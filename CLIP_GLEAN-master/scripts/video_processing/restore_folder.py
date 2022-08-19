# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import glob

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('dir_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save restoration result')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id') # do not use this option
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_paths = glob.glob(osp.join(args.dir_path, '*'))

    for i, img_path in enumerate(img_paths):
        output = restoration_inference(model, img_path)
        output = tensor2img(output)
        
        filename, extension = osp.splitext(osp.basename(img_path))
        mmcv.imwrite(output, osp.join(args.save_path, f'{filename}_glean{extension}'))

if __name__ == '__main__':
    main()