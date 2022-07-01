# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

import torchvision.transforms as T
import requests
from PIL import Image

import numpy as np

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.core.mask.structures import BitmapMasks
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # instantiate weights
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # create model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # model.init_weights()

    # create dummy data
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    transforms = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    img1 = transforms(image)
    img2 = transforms(image)

    img = torch.stack([img1, img2], dim=0)
    img_metas = [
        {
            "filename": "./drive/MyDrive/ConvNeXT MaskRCNN/COCO/val2017/000000039769.jpg",
            "ori_filename": "000000039769.jpg",
            "ori_shape": (480, 640, 3),
            "img_shape": (480, 640, 3),
            "pad_shape": (480, 640, 3),
            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "flip": False,
            "flip_direction": None,
            "img_norm_cfg": {
                "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                "to_rgb": True,
            },
        },
        {
            "filename": "./drive/MyDrive/ConvNeXT MaskRCNN/COCO/val2017/000000039769.jpg",
            "ori_filename": "000000039769.jpg",
            "ori_shape": (480, 640, 3),
            "img_shape": (704, 939, 3),
            "pad_shape": (704, 960, 3),
            "scale_factor": np.array([1.4671875, 1.4666667, 1.4671875, 1.4666667], dtype=np.float32),
            "flip": False,
            "flip_direction": None,
            "img_norm_cfg": {
                "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                "to_rgb": True,
            },
        },
    ]
    labels = dict()
    labels["gt_labels"] = [torch.tensor([65, 65]), torch.tensor([65, 65])]
    labels["gt_bboxes"] = [
        torch.tensor([[252.8191, 29.9796, 301.3282, 163.3882], [0.0000, 23.2408, 54.6913, 79.8369]]),
        torch.tensor([[348.0057, 109.9930, 402.2309, 259.2000], [623.7056, 102.4561, 810.5126, 165.7544]]),
    ]
    labels["gt_bboxes_ignore"] = None
    masks = np.random.rand(2, 480, 640)
    h = 480
    w = 640
    labels["gt_masks"] = [BitmapMasks(masks, h, w), BitmapMasks(masks, h, w)]
    
    # do a single forward_train forward pass with dummy data
    losses = model.forward_train(img,
                      img_metas,
                      gt_bboxes=labels["gt_bboxes"],
                      gt_labels=labels["gt_labels"],
                      gt_bboxes_ignore=None,
                      gt_masks=labels["gt_masks"],
                      proposals=None)

    print("Losses:", losses)


if __name__ == '__main__':
    main()