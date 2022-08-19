import random
import time
from os import path as osp
import os 
import glob
from torch.utils import data as data
#from torchvision.transforms.functional import normalize
from torchvision import transforms as T
from torch.nn.functional import interpolate
from torch import flip

import numpy as np
from PIL import Image

class CLIPSRIterAnnotationDataset(data.Dataset):
    def __init__(self, opt, test = False):
        super(CLIPSRIterAnnotationDataset, self).__init__()
        self.opt = opt
        self.test = test

        self.scale = opt['scale']

        self.gt_size = opt['gt_size']
        self.lq_size = opt['gt_size'] // opt['scale']

        self.captions_folder = opt['caption_path'] #original

        # self.gt_folder = opt['hq_path']
        self.gt_folder = osp.join(opt['root_path'], 'data256x256_resize_png')
        self.gt_paths = sorted(glob.glob(osp.join(self.gt_folder, f'{opt["image_id"]}.png')))
        
        self.lq_folder = osp.join(opt['root_path'], 'data16x16_png')
        self.lq_paths = sorted(glob.glob(osp.join(self.lq_folder, f'{opt["image_id"]}.png')))

        # should not use sort function for caption files due to format of files (0,1,2,3...10,11 .txt)
        # self.captions_paths = sorted(glob.glob(osp.join(self.captions_folder, f'*.txt')))

        self.caption = opt['caption']
        self.totensor = T.ToTensor()
        self.norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.flip = opt['use_hflip']
        assert len(self.gt_paths) > 0 and len(self.lq_paths) > 0


    def transform(self, im):
        im = self.totensor(im)[:3,:,:]
        im = self.norm(im)
        return im

    # TODO need to decide how to feed captions to model.
    def __getitem__(self, idx):
        lq_path, gt_path = self.lq_paths[idx], self.gt_paths[idx]
        
        img_lq, img_gt = Image.open(lq_path), Image.open(gt_path)

        #caption
        # caption_idx = idx
        # caption_path = os.path.join(self.captions_folder, f'{caption_idx}.txt')
        # with open(caption_path) as f:
        #     cap = f.readlines()
        # cap = [c.strip() for c in cap]
        # cap = cap[random.randrange(0, len(cap))]

        cap = self.caption
        
        img_lq, img_gt = self.transform(img_lq), self.transform(img_gt)

        if self.flip and random.random() < 0.5:
            img_lq, img_gt = flip(img_lq, dims=(2,)), flip(img_gt, dims=(2,))
            # img_lq, img_gt = img_lq[:, :, ::-1], img_gt[:, :, ::-1]

        return {
            'lq':img_lq, 
            'gt': img_gt, 
            'cap': cap,
            'lq_path': lq_path, 
            'gt_path': gt_path,
            # 'caption_path': caption_path,
        }

    def __len__(self):
        return len(self.gt_paths)
