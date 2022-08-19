import glob
from os import path as osp

import torch
from torch.utils import data as data
from torchvision import transforms as T
from PIL import Image


transform = T.Compose(
    [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def read_img_seq(img_paths):
    imgs = [transform(Image.open(img_path)) for img_path in img_paths]
    imgs = torch.stack(imgs, dim=0)

    return imgs


class FolderVFIDataset(data.Dataset):
    def __init__(self, opt):
        super(FolderVFIDataset, self).__init__()
        self.opt = opt

        folder_lq, folder_gt = opt['folder_lq'], opt['folder_gt']

        self.cache_data = opt['cache_data']

        self.num_input_frames = opt['num_input_frames']

        # specifying given data type
        self.is_gt_given = folder_gt is not None
        self.is_lq_given = folder_lq is not None

        assert self.is_gt_given or self.is_lq_given, 'valid data path should be given'
        
        # NOTE: online resizing gt not implemented yet

        if self.is_gt_given:
            self.img_paths_gt = sorted(glob.glob(osp.join(folder_gt, '*')))
        if self.is_lq_given:
            self.img_paths_lq = sorted(glob.glob(osp.join(folder_lq, '*')))
            
        if self.cache_data:
            if self.is_lq_given:
                self.imgs_lq = read_img_seq(self.img_paths_lq)
            if self.is_gt_given:
                self.imgs_gt = read_img_seq(self.img_paths_gt)
        else:
            if self.is_lq_given:
                self.imgs_lq = self.img_paths_lq
            if self.is_gt_given:
                self.imgs_gt = self.img_paths_gt

        print(f'test dataset includes {len(self.imgs_lq)} number of frames')

    def __getitem__(self, idx):
        indices = list(range(idx, idx + self.num_input_frames))

        if self.is_gt_given:
            if self.cache_data:
                imgs_gt = self.imgs_gt.index_select(0, torch.LongTensor(indices))
            else:
                img_paths_gt = [self.imgs_gt[i] for i in indices]
                imgs_gt = read_img_seq(img_paths_gt)
            imgs_gt.squeeze_(0)
            gt_path = [self.img_paths_gt[i] for i in indices]

        else:
            gt_path, imgs_gt = 0 ,0 #TODO: cleanup

        if self.is_lq_given:
            if self.cache_data:
                imgs_lq = self.imgs_lq.index_select(0, torch.LongTensor(indices))
            else:
                img_paths_lq = [self.imgs_lq[i] for i in indices]
                imgs_lq = read_img_seq(img_paths_lq)
            imgs_lq.squeeze_(0)
            lq_path = [self.img_paths_lq[i] for i in indices]

        else:
            lq_path, imgs_lq = 0,0

        return {
            'lq':imgs_lq, 
            'gt': imgs_gt, 
            'lq_path': lq_path, 
            'gt_path': gt_path,
            'is_lq_given': self.is_lq_given,
            'is_gt_given': self.is_gt_given
        }

    def __len__(self):
        return len(self.imgs_lq) - self.num_input_frames + 1
