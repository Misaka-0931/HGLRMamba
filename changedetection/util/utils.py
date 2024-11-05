import numpy as np
import logging
import os
import os
import torch
import torch.distributed as dist
from timm.utils.model import unwrap_model, get_state_dict

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.lst_num_per_cls = np.zeros((num_classes,))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def _get_num_per_class(self, label_true):
        res = []
        for i in range(0, self.num_classes):
            res.append((label_true==i).sum())
        return np.array(res)

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            self.lst_num_per_cls += self._get_num_per_class(lt.flatten())

    def evaluate(self, flag_cls_weight=False, flag_ignore_background=False):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        if flag_cls_weight:
            if flag_ignore_background:
                tmp_weight = self.lst_num_per_cls
                tmp_weight[0] = 0
                tmp_weight = tmp_weight / (tmp_weight.sum())
            else:
                tmp_weight = self.lst_num_per_cls / (self.lst_num_per_cls.sum())
            return iu, np.nansum(iu * tmp_weight)
        else:
            return iu, np.nanmean(iu)

    def calculate_f1_and_recall(self, flag_ignore_background=False):
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        f1 = 2 * (precision * recall) / (precision + recall)

        if flag_ignore_background:
            recall = recall[1:]
            precision = precision[1:]
            f1 = f1[1:]

        return recall, f1


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    max_accuracy_e = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']
        max_accuracy_e = checkpoint['max_accuracy_e']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_e

def load_pretrained(ckpt_path, model, logger):
    logger.info(f"==============> Loading pretrained form {ckpt_path}....................")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # msg = model.load_pretrained(checkpoint['model'])
    # logger.info(msg)
    # logger.info(f"=> Loaded successfully {ckpt_path} ")
    # del checkpoint
    # torch.cuda.empty_cache()
    state_dict = checkpoint['state_dict_ema'] if 'state_dict_ema' in checkpoint.keys() else checkpoint

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # linear interpolate agent bias if h/w not match, bicubic interpolate agent bias if agent_num not match
    agent_bias_keys = [k for k in state_dict.keys() if ("ah_bias" in k) or ("aw_bias" in k)
                                or ("ha_bias" in k) or ("wa_bias" in k)]
    for k in agent_bias_keys:
        if "ah_bias" in k:
            squeeze_dim, permute = -1, False
        elif "aw_bias" in k:
            squeeze_dim, permute = -2, False
        elif "ha_bias" in k:
            squeeze_dim, permute = -2, True
        else:
            squeeze_dim, permute = -3, True
        agent_bias_pretrained = state_dict[k].squeeze(dim=0).squeeze(dim=squeeze_dim)
        agent_bias_current = model.state_dict()[k].squeeze(dim=0).squeeze(dim=squeeze_dim)
        if permute:
            agent_bias_pretrained = agent_bias_pretrained.permute(0, 2, 1)
            agent_bias_current = agent_bias_current.permute(0, 2, 1)
        num_heads1, agent_num1, hw1 = agent_bias_pretrained.size()
        num_heads2, agent_num2, hw2 = agent_bias_current.size()
        if num_heads1 != num_heads2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if agent_num1 != agent_num2:
                a1 = int(agent_num1 ** 0.5)
                a2 = int(agent_num2 ** 0.5)
                agent_bias_pretrained_resized = agent_bias_pretrained.permute(0, 2, 1).reshape(num_heads1, hw1, a1, a1)
                agent_bias_pretrained_resized = torch.nn.functional.interpolate(
                    agent_bias_pretrained_resized, size=(a2, a2), mode='bicubic').flatten(2).permute(0, 2, 1)
                agent_bias_pretrained = agent_bias_pretrained_resized
                if permute:
                    agent_bias_pretrained_resized = agent_bias_pretrained_resized.permute(0, 2, 1)
                state_dict[k] = agent_bias_pretrained_resized.unsqueeze(dim=0).unsqueeze(dim=squeeze_dim)
            if hw1 != hw2:
                # linear interpolate agent bias if not match
                agent_bias_pretrained_resized = torch.nn.functional.interpolate(
                    agent_bias_pretrained, size=hw2, mode='linear')
                if permute:
                    agent_bias_pretrained_resized = agent_bias_pretrained_resized.permute(0, 2, 1)
                state_dict[k] = agent_bias_pretrained_resized.unsqueeze(dim=0).unsqueeze(dim=squeeze_dim)

    agent_bias_keys = [k for k in state_dict.keys() if ("an_bias" in k) or ("na_bias" in k)]
    for k in agent_bias_keys:
        agent_bias_pretrained = state_dict[k]
        agent_bias_current = model.state_dict()[k]
        num_heads1, agent_num1, h1, w1 = agent_bias_pretrained.size()
        num_heads2, agent_num2, h2, w2 = agent_bias_current.size()
        if (num_heads1 != num_heads2) or (h1 != h2) or (w1 != w2):
            print(f"Error in loading {k}, passing......")
        else:
            if agent_num1 != agent_num2:
                a1 = int(agent_num1 ** 0.5)
                a2 = int(agent_num2 ** 0.5)
                agent_bias_pretrained_resized = agent_bias_pretrained.flatten(2).permute(0, 2, 1).reshape(num_heads1, -1, a1, a1)
                agent_bias_pretrained_resized = torch.nn.functional.interpolate(
                    agent_bias_pretrained_resized, size=(a2, a2), mode='bicubic').flatten(2).permute(0, 2, 1)
                state_dict[k] = agent_bias_pretrained_resized.reshape(num_heads2, agent_num2, h2, w2)

    # bicubic interpolate patch_embed.proj if not match
    patch_embed_keys = [k for k in state_dict.keys() if ("patch_embed" in k) and (".proj.weight" in k)]
    for k in patch_embed_keys:
        patch_embed_pretrained = state_dict[k]
        patch_embed_current = model.state_dict()[k]
        out1, in1, h1, w1 = patch_embed_pretrained.size()
        out2, in2, h2, w2 = patch_embed_current.size()
        if (out1 != out2) or (in1 != in2):
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if (h1 != h2) or (w1 != w2):
                # bicubic interpolate patch_embed.proj if not match
                patch_embed_pretrained_resized = torch.nn.functional.interpolate(
                    patch_embed_pretrained, size=(h2, w2), mode='bicubic')
                state_dict[k] = patch_embed_pretrained_resized

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                i, j = L1 - S1 ** 2, L2 - S2 ** 2
                absolute_pos_embed_pretrained_ = absolute_pos_embed_pretrained[:, i:, :].reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained_ = absolute_pos_embed_pretrained_.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained_, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = torch.cat([absolute_pos_embed_pretrained[:, :j, :],
                                           absolute_pos_embed_pretrained_resized], dim=1)

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{ckpt_path}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, max_accuracy_e, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  # 'model_ema': model_ema.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'max_accuracy_e': max_accuracy_e,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_checkpoint_ema(config, epoch, model, model_ema, max_accuracy, max_accuracy_e, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  # 'model_ema': model_ema.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'max_accuracy_e': max_accuracy_e,
                  'epoch': epoch,
                  'config': config}
    save_state['state_dict_ema'] = get_state_dict(model_ema, unwrap_model)

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_checkpoint_ema_new(config, epoch, model, model_ema, max_accuracy, max_accuracy_e, optimizer, lr_scheduler, logger, name=None):
    save_state = {'model': model.state_dict(),
                  # 'model_ema': model_ema.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'max_accuracy_e': max_accuracy_e,
                  'epoch': epoch,
                  'config': config}
    save_state['state_dict_ema'] = get_state_dict(model_ema, unwrap_model)

    if name==None:
        old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch-3}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

    if name!=None:
        save_path = os.path.join(config.OUTPUT, f'{name}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
