"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import errno
import os
import sys
import time
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'mixup_data', 'mixup_criterion',
           'weight_filler', 'colorEncode', 'RandomPixelJitter', 'RandomErasing']


class RandomPixelJitter(object):
    def __init__(self, range):
        self.range = range
        assert len(range) == 2

    def __call__(self, im):
        pic = np.array(im)
        noise = np.random.randint(self.range[0], self.range[1], pic.shape[-1])
        pic = pic + noise
        pic = pic.astype(np.uint8)
        return Image.fromarray(pic)
    
    
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean    # erasing mean value
        self.sl = sl    # min erasing area
        self.sh = sh    # max erasing area
        self.r1 = r1    # min aspect ratio

    def __call__(self, im):
        if random.uniform(0, 1) > self.probability:
            return im

        pic = np.array(im)
        h, w = pic.shape[:2]
        target_area = random.uniform(self.sl, self.sh) * (h * w)
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        xx = min(int(round(math.sqrt(target_area * aspect_ratio))), w - 2)
        yy = min(int(round(math.sqrt(target_area / aspect_ratio))), h - 2)
        x1 = random.randint(0, w - xx)
        y1 = random.randint(0, h - yy)

        if len(pic.shape) == 3:
            pic[y1:y1 + yy, x1:x1 + xx, 0] = self.mean[0]
            pic[y1:y1 + yy, x1:x1 + xx, 1] = self.mean[1]
            pic[y1:y1 + yy, x1:x1 + xx, 2] = self.mean[2]
        else:
            pic[y1:y1 + yy, x1:x1 + xx] = self.mean[0]

        pic = pic.astype(np.uint8)
        return Image.fromarray(pic)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def weight_filler(src, dst):
    updated_dict = dst.copy()
    match_layers = []
    mismatch_layers = []
    for dst_k in dst:
        if dst_k in src:
            src_k = dst_k
            if src[src_k].shape == dst[dst_k].shape:
                match_layers.append(dst_k)
                updated_dict[dst_k] = src[src_k]
            else:
                mismatch_layers.append(dst_k)
        elif dst_k.replace('module.', '') in src:
            src_k = dst_k.replace('module.', '')
            if src[src_k].shape == dst[dst_k].shape:
                match_layers.append(dst_k)
                updated_dict[dst_k] = src[src_k]
            else:
                mismatch_layers.append(dst_k)
        elif 'module.' + dst_k in src:
            src_k = 'module.' + dst_k
            if src[src_k].shape == dst[dst_k].shape:
                match_layers.append(dst_k)
                updated_dict[dst_k] = src[src_k]
            else:
                mismatch_layers.append(dst_k)
        else:
            mismatch_layers.append(dst_k)

    return updated_dict, match_layers, mismatch_layers


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(label, color_map):
    label = label.astype('int')
    color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    category = np.unique(label)
    for c in list(category):
        color[np.where(label == c)] = color_map[c]

    return color


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
