import glob
import os.path as osp

import cv2
import mmcv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

import sys
sys.path.append('../scripts/image_processing/ResizeRight')
sys.path.append('scripts/image_processing/ResizeRight')
from scripts.image_processing.ResizeRight.resize_right import resize
from PIL import Image

from torchvision import transforms as T

class HugeDataset(Dataset):
    def __init__(self, opt, is_train=True):
        super(HugeDataset, self).__init__()

        self.opt = opt

        assert is_train, 'Test mode not supported for this dataset code'

        datapath = '/data/jinsuyoo/TalkingHead-1KH/train/cropped_clips'

        self.num_gt_frames = opt['num_gt_frames']
        self.num_lq_frames = self.num_gt_frames // 2 + 1

        self.scale = opt['scale']
        self.use_hflip = opt['use_hflip']

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

        self.videolist = sorted(glob.glob(osp.join(datapath, '*.mp4')))
        print(f'{len(self.videolist)} number of training video exists!')

        gt_folder_celebahq = osp.join(
            opt['root_path'], 'CelebAHQ', f'images1024x1024')
        gt_paths_celebahq = sorted(
            glob.glob(osp.join(gt_folder_celebahq, f'*.png')))
        #lq_folder_celebahq = osp.join(
        #    opt['root_path'], 'CelebAHQ', f'images{self.lq_size}x{self.lq_size}')
        #lq_paths_celebahq = sorted(glob.glob(osp.join(lq_folder_celebahq, f'*.png')))

        gt_folder_ffhq = osp.join(
            opt['root_path'], 'FFHQ', f'images1024x1024')
        gt_paths_ffhq = sorted(
            glob.glob(osp.join(gt_folder_ffhq, f'*.png')))
        #lq_folder_ffhq = osp.join(
        #    opt['root_path'], 'FFHQ', f'images{self.lq_size}x{self.lq_size}')
        #lq_paths_ffhq = sorted(glob.glob(osp.join(lq_folder_ffhq, f'*.png')))

        self.gt_paths_image = gt_paths_celebahq + gt_paths_ffhq
        #self.lq_paths_image = lq_paths_celebahq + lq_paths_ffhq
        print(f'{len(self.gt_paths_image)} number of trainig image exists!')
        self.transform_train = T.Compose([
            T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def img2tensor(self, imgs, bgr2rgb, float32):
        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            return img

        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        else:
            return _totensor(imgs, bgr2rgb, float32)

    def __getitem__(self, idx):
        video_path = self.videolist[idx]
        video_reader = mmcv.VideoReader(video_path)
        
        random_frame_indices = random.randint(
            0, len(video_reader) - self.num_gt_frames)
        random_frame_indices = list(
            range(random_frame_indices, random_frame_indices + self.num_gt_frames))
        
        #######  get images
        random_image_indices = random.sample(
            self.gt_paths_image, self.num_gt_frames)
        #images_lq, images_gt = [], []
        images_gt = []
        for image_idx in random_image_indices:
            #image_lq = Image.open(self.lq_paths_image[image_idx])
            image_gt = Image.open(self.gt_paths_image[image_idx])
            #image_lq = self.transform_train(image_lq)
            image_gt = self.transform_train(image_gt)
            #images_lq.append(image_lq)
            images_gt.append(image_gt)
        #images_lq = torch.stack(images_lq)
        images_gt = torch.stack(images_gt)

        if not self.use_hflip and random.random() < 0.5:
            #images_lq = images_lq[..., ::-1]
            images_gt = images_gt[..., ::-1]

        #######

        frames = [video_reader[i] for i in random_frame_indices]

        frames = self.img2tensor(frames, bgr2rgb=True, float32=True)
        frames = torch.stack(frames)

        if not self.use_hflip and random.random() < 0.5:
            frames = frames[..., ::-1]
        frames = torch.clamp(frames, 0, 255) / 255

        # generate GT frames
        frames_gt = torch.clamp(resize(frames, out_shape=(512, 512)), 0, 1)

        # generate LQ frames
        frames_lq = (frames_gt * 255).round().type(torch.uint8) / 255 # requantize
        frames_lq = torch.clamp(
            resize(frames_lq, out_shape=(512//self.scale, 512//self.scale)), 0, 1)

        # normalize
        normalize(frames_gt, self.mean, self.std, inplace=True)
        normalize(frames_lq, self.mean, self.std, inplace=True)
        
        # sparse sampling for temporal SR
        indices = torch.tensor([i*2 for i in range(self.num_gt_frames//2 + 1)])
        frames_lq = torch.index_select(frames_lq, 0, indices)

        if self.num_gt_frames == 1:
            # (1, 3, h, w) -> (3, h, w)
            frames_gt.squeeze_(0), frames_lq.squeeze_(0)

        return {
            'gt': frames_gt, 
            'lq': frames_lq, 
            'gt_path': video_path,
            'fps': video_reader.fps,
            'gt_images': images_gt,
            #'lq_images': images_lq
        }

    def __len__(self):
        return len(self.videolist)