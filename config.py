# --------------------------------------------------------
# Copyright (c) 2015 BUPT-Priv
# Licensed under The MIT License [see LICENSE for details]
# Written by Yang Lu
# --------------------------------------------------------

import os
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import numpy as np

cur_pth = os.getcwd()

__C = edict()
cfg = __C

#
# Global options
#
__C.workers = 8  # number of data loading workers
__C.pixel_mean = [0.485, 0.456, 0.406]  # mean value of imagenet
__C.pixel_std = [0.229, 0.224, 0.225]  # std value of imagenet
__C.rng_seed = 3  # manual seed
__C.eps = 1e-14  # A very small number that's used many times
__C.eps5 = 1e-5  # A small number that's used many times
__C.gpu_ids = '0'  # gpu_ids, like: 0,1,2,3

#
# Classification options
#
__C.CLS = edict()
__C.CLS.arch = 'resnet18'  # model architecture
__C.CLS.data_root = '~/Database/ILSVRC2017/Data/CLS-LOC/'  # path to dataset
__C.CLS.train_folder = 'train'  # train folder
__C.CLS.val_folder = 'val'  # val folder
__C.CLS.epochs = 100  # number of total epochs to run
__C.CLS.train_batch = 256  # train batchsize of all gpus
__C.CLS.test_batch = 200  # test batchsize
__C.CLS.base_lr = 0.1  # base learning rate
__C.CLS.lr_schedule = [30, 60]  # decrease learning rate at these epochs
__C.CLS.gamma = 0.1  # base_lr is multiplied by gamma on lr_schedule
__C.CLS.momentum = 0.9  # momentum
__C.CLS.weight_decay = 1e-4  # weight_decay
__C.CLS.fix_bn = False  # fix bn params
__C.CLS.num_classes = 1000  # number of classes
__C.CLS.base_size = 256  # base size
__C.CLS.crop_size = 224  # crop size
__C.CLS.rotation = []  # list, randomly rotate the image by angle, etc. [-10, 10]
__C.CLS.pixel_jitter = []  # list, random pixel jitter, etc. [-20, 20]
__C.CLS.grayscale = 0  # float, randomly convert image to gray-scale with a probability, etc. 0.1
__C.CLS.random_erasing = False  # using random erasing data augmentation
__C.CLS.disp_iter = 20  # display iteration
__C.CLS.ckpt = 'ckpts/imagenet/resnet18/'  # path to save checkpoint
__C.CLS.resume = ''  # path to latest checkpoint
__C.CLS.start_epoch = 0  # manual epoch number (useful on resume)
__C.CLS.pretrained = ''  # path to pretrained model
__C.CLS.cosine_lr = False  # using cosine learning rate
__C.CLS.validate = True  # validate
__C.CLS.evaluate = False  # evaluate


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
