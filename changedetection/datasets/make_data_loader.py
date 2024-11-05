import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import changedetection.datasets.imutils as imutils

def img_loader(path):
    return np.array(Image.open(path), np.float32)


def one_hot_encoding(image, num_classes=8):

    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, data_path, crop_size, type='train', data_loader=img_loader, logger=None):
        self.data_path = data_path
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.crop_size = crop_size
        data_list_path = data_path + '/list/' + type + '.txt'
        if logger is not None:
            logger.info(f"Data Path is : {data_list_path}")
        with open(data_list_path, "r") as f:
            self.data_list = [data_name.strip() for data_name in f]

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.data_path, 'A', self.data_list[index])
        post_path = os.path.join(self.data_path, 'B', self.data_list[index])
        label_path = os.path.join(self.data_path, 'label', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, logger=None,**kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU-CD' in args.dataset or 'LEVIR-CD' or 'S2Looking-CD' or "DSIFN-CD" in args.dataset:
        train_dataset = ChangeDetectionDatset(args.data_path, args.crop_size, type='train',logger=logger)
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs, num_workers=16,
                                 drop_last=False, sampler=train_sampler)
        test_dataset = ChangeDetectionDatset(args.data_path, args.crop_size, type='test',logger=logger)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs, num_workers=16,
                                 drop_last=False, sampler=test_sampler)
        return train_data_loader, test_data_loader

    else:
        raise NotImplementedError

def make_test_data_loader(args, logger=None,**kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU-CD' in args.dataset or 'LEVIR-CD' or 'S2Looking-CD' or "DSIFN-CD" in args.dataset:

        test_dataset = ChangeDetectionDatset(args.data_path, args.crop_size, type='test',logger=logger)
        test_data_loader = DataLoader(test_dataset, batch_size=16, **kwargs, num_workers=16,
                                 drop_last=False)
        return test_data_loader

    else:
        raise NotImplementedError
