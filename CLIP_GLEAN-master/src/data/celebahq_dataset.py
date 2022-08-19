import random
import time
from os import path as osp
import glob
from torch.utils import data as data
#from torchvision.transforms.functional import normalize
from torchvision import transforms as T
from torch.nn.functional import interpolate

import numpy as np
from PIL import Image


class CelebAHQDataset(data.Dataset):
    def __init__(self, opt):
        super(CelebAHQDataset, self).__init__()
        self.opt = opt

        self.scale = opt['scale']

        self.gt_size = opt['gt_size']
        self.lq_size = opt['gt_size'] // opt['scale']
        self.gt_folder = osp.join(opt['root_path'], 'CelebAHQ', f'images{self.gt_size}x{self.gt_size}')
        self.gt_paths = sorted(glob.glob(osp.join(self.gt_folder, f'*.png')))
        if self.gt_size == 512:
            self.lq_folder = osp.join(opt['root_path'], 'CelebAHQ', f'images{self.lq_size}x{self.lq_size}_from512')
        else:
            self.lq_folder = osp.join(opt['root_path'], 'CelebAHQ', f'images{self.lq_size}x{self.lq_size}')
        self.lq_paths = sorted(glob.glob(osp.join(self.lq_folder, f'*.png')))

        self.lq_paths, self.gt_paths = self.lq_paths[:100], self.gt_paths[:100]

        self.transform = T.Compose([
            T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        lq_path, gt_path = self.lq_paths[idx], self.gt_paths[idx]
        
        img_lq, img_gt = Image.open(lq_path), Image.open(gt_path)
        
        img_lq, img_gt = self.transform(img_lq), self.transform(img_gt)

        return {
            'lq':img_lq, 
            'gt': img_gt, 
            'lq_path': lq_path, 
            'gt_path': gt_path,
            'is_lq_given': True,
            'is_gt_given': True
        }

    def __len__(self):
        return len(self.gt_paths)
