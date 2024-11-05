import os
import time
import argparse
import datetime
import numpy as np
import sys
sys.path.append('/data1/wuxiaomeng/code/HGLRMamba')
print(sys.path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from timm.utils import accuracy, AverageMeter, ModelEma
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from configs.config import get_config
from models.build import build_model
from datasets.make_data_loader import make_data_loader
from util.lr_scheduler import build_scheduler
from util.optimizer import build_optimizer
from util.logger import create_logger
from util.utils import get_grad_norm, reduce_tensor
import utils_func.lovasz_loss as L
from utils_func.dice_loss import DiceLoss, make_one_hot
from utils_func.metrics import Evaluator
import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Harmonize Global and Local Representations for Remote Sensing Change Detection', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')

    parser.add_argument('--dataset', type=str, default='LEVIR-CD')

    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable_amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')
    parser.add_argument('--find-unused-params', action='store_true', default=False)

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def main():
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    args, config = parse_option()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True


    config.defrost()
    config.LOCAL_RANK = local_rank
    config.freeze()



    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        ckpt_path = os.path.join(config.OUTPUT, "ckpts")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)


    # print config
    logger.info(config.dump())
    data_loader_train, data_loader_test = make_data_loader(args, logger=logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    if dist.get_rank() == 0:
        inputs = (torch.randn(2, 3, 256, 256).cuda(),torch.randn(2, 3, 256, 256).cuda())
        logger.info(flop_count_str(FlopCountAnalysis(model, inputs)))
        del inputs
    optimizer = build_optimizer(config, model)

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True,
    #                                             find_unused_parameters=True)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True,)

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    total_epochs = config.TRAIN.EPOCHS


    best_kc = best_epoch = -1
    best_round = []
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config=config, model=model, data_loader=data_loader_train, \
                        optimizer=optimizer, epoch=epoch, total_epochs=total_epochs, \
                    lr_scheduler=lr_scheduler, logger=logger)

        rec, pre, oa, f1_score, iou, kc = validate(epoch=epoch + 1,
                                                   data_loader=data_loader_test,
                                                   model=model,
                                                   logger=logger)

        if best_kc < kc:
            best_epoch = epoch + 1
            best_kc = kc
            best_round = [rec, pre, oa, f1_score, iou, kc]

            if rank == 0:
                save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
                torch.save(save_state, ckpt_path + '/best.pth')


        logger.info(f"Previous best epoch is {best_epoch}/{total_epochs}\t,"
                    f'Racall rate is {best_round[0]},\t'
                    f'Precision rate is {best_round[1]},\t'
                    f"OA is {best_round[2]}, F1 score is {best_round[3]},\t"
                    f"IoU is {best_round[4]},\t "
                    f"Kappa coefficient is {best_round[-1]}")

        if rank == 0:
            save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
            torch.save(save_state, ckpt_path + '/last.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config,
                    model,
                    data_loader,
                    optimizer,
                    epoch,
                    lr_scheduler,
                    logger,
                    total_epochs,
                    mixup_fn=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    # dice = DiceLoss(ignore_index=255)
    start = time.time()
    end = time.time()

    scaler = GradScaler()

    for idx, (pre_change_imgs, post_change_imgs, targets, _) in enumerate(data_loader):
        optimizer.zero_grad()

        pre_change_imgs = pre_change_imgs.cuda(non_blocking=True)
        post_change_imgs = post_change_imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).long()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.ENABLE_AMP:
            with autocast():

                outputs = model(pre_change_imgs, post_change_imgs)
                ce_loss_1 = F.cross_entropy(outputs, targets, ignore_index=255)
                outputs_softmax = F.softmax(outputs, dim=1)
                lovasz_loss = L.lovasz_softmax(outputs_softmax, targets, ignore=255)
                main_loss = ce_loss_1 +  0.75 * lovasz_loss
                final_loss = main_loss
                # final_loss.backward()
            scaler.scale(final_loss).backward()

            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(pre_change_imgs, post_change_imgs)
            ce_loss_1 = F.cross_entropy(outputs, targets, ignore_index=255)
            outputs_softmax = F.softmax(outputs, dim=1)
            lovasz_loss = L.lovasz_softmax(outputs_softmax, targets, ignore=255)
            main_loss = ce_loss_1 + 0.75 * lovasz_loss
            final_loss = main_loss

            final_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()

        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()

        # assert torch.isnan(final_loss).sum() == 0, print(final_loss)
        loss_meter.update(final_loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(data_loader, model, logger, epoch):
    model.eval()
    logger.info(f"Start Test: {epoch} ")
    evaluator = Evaluator(num_class=2)
    start = time.time()
    # torch.cuda.empty_cache()
    for itera, data in enumerate(data_loader):
        pre_change_imgs, post_change_imgs, labels, _ = data
        pre_change_imgs = pre_change_imgs.cuda().float()
        post_change_imgs = post_change_imgs.cuda()
        labels = labels.cuda().long()

        output_1 = model(pre_change_imgs, post_change_imgs)

        output_1 = output_1.data.cpu().numpy()
        output_1 = np.argmax(output_1, axis=1)
        labels = labels.cpu().numpy()

        evaluator.add_batch(labels, output_1)


    end = time.time()

    f1_score = evaluator.Pixel_F1_score()
    oa = evaluator.Pixel_Accuracy()
    rec = evaluator.Pixel_Recall_Rate()
    pre = evaluator.Pixel_Precision_Rate()
    iou = evaluator.Intersection_over_Union()
    kc = evaluator.Kappa_coefficient()

    reduce_f1 = torch.tensor(f1_score).cuda()
    reduce_oa = torch.tensor(oa).cuda()
    reduce_rec = torch.tensor(rec).cuda()
    reduce_pre = torch.tensor(pre).cuda()
    reduce_iou = torch.tensor(iou).cuda()
    reduce_kc = torch.tensor(kc).cuda()

    reduce_f1 = reduce_tensor(reduce_f1)
    reduce_oa = reduce_tensor(reduce_oa)
    reduce_rec = reduce_tensor(reduce_rec)
    reduce_pre = reduce_tensor(reduce_pre)
    reduce_iou = reduce_tensor(reduce_iou)
    reduce_kc = reduce_tensor(reduce_kc)

    logger.info(f'Test {epoch}:, \t'
                f'Racall rate is {reduce_rec.cpu().numpy():.4f}, \t'
                f'Precision rate is {reduce_pre.cpu().numpy():.4f}, \t'
                f' OA is {reduce_oa.cpu().numpy():.4f},'
               f'F1 score is {reduce_f1.cpu().numpy():.4f}, \t'
               f'IoU is {reduce_iou.cpu().numpy():.4f}, \t'
               f'Kappa coefficient is {reduce_kc.cpu().numpy():.4f}'
            )
    logger.info(
        f'Test time is {datetime.timedelta(seconds=int(end - start))}'
    )
    return reduce_rec.item(), reduce_pre.item(), reduce_oa.item(), reduce_f1.item(), reduce_iou.item(), reduce_kc.item()



if __name__ == '__main__':

    main()
