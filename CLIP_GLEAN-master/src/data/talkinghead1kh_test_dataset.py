import glob
import os.path as osp
import os
import cv2
import mmcv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
from torchvision import transforms as T

import sys
sys.path.append('../scripts/image_processing/ResizeRight')
sys.path.append('scripts/image_processing/ResizeRight')
from scripts.image_processing.ResizeRight.resize_right import resize
from PIL import Image


def generate_talkinghead_datasets(opt) -> list:
    gt_root, lq_root = opt['dataroot_gt'], opt['dataroot_lq']

    subfolders_lq = sorted(glob.glob(osp.join(lq_root, '*')))
    subfolders_gt = sorted(glob.glob(osp.join(gt_root, '*')))
        
    subfolders_lq = subfolders_lq[:opt['num_val_clip']]
    subfolders_gt = subfolders_gt[:opt['num_val_clip']]

    datasets = []
    for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
        dataset = TalkingHead1KHSingleClipDataset(
            subfolder_lq=subfolder_lq, subfolder_gt=subfolder_gt, opt=opt)
        datasets.append(dataset)

    return datasets


transform = T.Compose(
    [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def read_img_seq(img_paths):
    imgs = [transform(Image.open(img_path)) for img_path in img_paths]
    imgs = torch.stack(imgs, dim=0)

    return imgs

def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.
    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]
    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



class TalkingHead1KHSingleClipDataset(Dataset):
    def __init__(self, subfolder_lq, subfolder_gt, opt):
        super(TalkingHead1KHSingleClipDataset, self).__init__()

        self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        self.cache_data = opt['cache_data']

        self.opt = opt

        self.num_gt_frames = opt['num_gt_frames']
        self.num_lq_frames = self.num_gt_frames // 2 + 1

        # get frame list for lq and gt
        subfolder_name = osp.basename(subfolder_lq)
        
        #img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
        #img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))
        self.img_paths_lq = sorted(glob.glob(osp.join(subfolder_lq, '*')))
        self.img_paths_gt = sorted(glob.glob(osp.join(subfolder_gt, '*')))

        self.img_paths_lq = self.img_paths_lq[:opt['num_val_frame_per_clip']]
        self.img_paths_gt = self.img_paths_gt[:opt['num_val_frame_per_clip']]
        
        # gt frames should odd number
        if len(self.img_paths_gt) % 2 == 0:
            self.img_paths_gt = self.img_paths_gt[:-1]
            self.img_paths_lq = self.img_paths_lq[:-1]
        
        # cache data or save the frame list
        if self.cache_data:
            print(f'Cache clip [{subfolder_name}] for VideoTestDataset')
            self.imgs_lq = read_img_seq(self.img_paths_lq)
            self.imgs_gt = read_img_seq(self.img_paths_gt)
        else:
            self.imgs_lq = self.img_paths_lq
            self.imgs_gt = self.img_paths_gt
    
        # for vfi
        #self.num_gt_frames = opt['num_gt_frames']
        #self.num_lq_frames = self.num_gt_frames // 2 + 1

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

    def imfrombytes(self, content, flag='color', float32=False):
        """Read an image from bytes.
        Args:
            content (bytes): Image bytes got from files or other streams.
            flag (str): Flags specifying the color type of a loaded image,
                candidates are `color`, `grayscale` and `unchanged`.
            float32 (bool): Whether to change to float32., If True, will also norm
                to [0, 1]. Default: False.
        Returns:
            ndarray: Loaded image array.
        """
        img_np = np.frombuffer(content, np.uint8)
        imread_flags = {
            'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
        img = cv2.imdecode(img_np, imread_flags[flag])
        if float32:
            img = img.astype(np.float32) / 255.
        return img

    def __getitem__(self, index):
        if not self.num_gt_frames == self.num_lq_frames:
            index *= 2
        indices = list(range(index, index + self.num_gt_frames))
        
        lq_paths = [self.img_paths_lq[i] for i in indices]
        gt_paths = [self.img_paths_gt[i] for i in indices]

        if self.cache_data:
            imgs_lq = self.imgs_lq.index_select(0, torch.LongTensor(indices))
            imgs_gt = self.imgs_gt.index_select(0, torch.LongTensor(indices))
        else:
            imgs_lq = read_img_seq(self.img_paths_lq)
            imgs_gt = read_img_seq(self.img_paths_gt)

        # sparse sampling for temporal SR
        sparse_indices = torch.tensor([i*2 for i in range(self.num_gt_frames//2 + 1)])
        imgs_lq = torch.index_select(imgs_lq, 0, sparse_indices)
        lq_paths = [lq_paths[i] for i in sparse_indices]

        if self.num_gt_frames == 1:
            imgs_gt.squeeze_(0), imgs_lq.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': imgs_gt, #img_gt,  # (c, h, w)
            'lq_path': lq_paths,  # lq paths # center frame
            'gt_path': gt_paths,
            'is_lq_given': True,
            'is_gt_given': True
        }

    def __len__(self):
        if self.num_gt_frames == self.num_lq_frames:
            return len(self.img_paths_lq)
        else:
            return len(self.img_paths_gt) // 2
        #return len(self.img_paths_lq) - self.num_lq_frames + 1
        
    '''
    def __getitem__(self, idx):
        video_path = self.videolist[idx]

        video_reader = mmcv.VideoReader(video_path)

        num_frames, fps, h, w = len(video_reader), video_reader.fps, video_reader.height, video_reader.width
        
        random_frame_indices = random.randint(0, num_frames-self.num_gt_frames)
        random_frame_indices = list(
            range(random_frame_indices, random_frame_indices + self.num_gt_frames))
        
        frames = [video_reader[i] for i in random_frame_indices]
        #frames = np.stack(frames, axis=0)

        # bgr. (t h w c) (0~255) (uint8) 
        #frames = frames.transpose(0, 3, 1, 2)
        #frames = (frames / 255).astype(np.float32)

        # numpy to tensor
        #frames = [self.transform(frames[i]) for i in range(frames.shape[0])]
        #frames = np.stack(frames, axis=0)

        # tensor 
        #frames = frames[:, :, :, ::-1] # bgr to rgb
        #frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))

        frames = self.img2tensor(frames, bgr2rgb=True, float32=True)
        frames = torch.stack(frames)

        if not self.use_flip and random.random() < 0.5:
            frames = frames[..., ::-1]

        frames = torch.clamp(frames, 0, 255) / 255

        frames_gt = torch.clamp(resize(frames, out_shape=(512, 512)), 0, 1)
        frames_lq = torch.clamp(resize(frames_gt, out_shape=(32, 32)), 0, 1)

        # normalize
        normalize(frames_gt, self.mean, self.std, inplace=True)
        normalize(frames_lq, self.mean, self.std, inplace=True)
        
        # sparse sampling for vfi
        indices = torch.tensor([i*2 for i in range(self.num_gt_frames//2 + 1)])
        frames_lq = torch.index_select(frames_lq, 0, indices)

        if self.num_gt_frames == 1:
            #print('here')
            #print(frames_gt.shape)
            frames_gt, frames_lq = frames_gt.squeeze(0), frames_lq.squeeze(0)
            #print(frames_gt.shape)

        return {
            'gt': frames_gt, 
            'lq': frames_lq, 
            'gt_path': video_path
        }

    def __len__(self):
        return len(self.videolist)
    '''









class TalkingHead1KHTestDataset(Dataset):
    """
    TalkingHead1KHDataset contains 500k 1minute cropped clips
    we 
    1) open clip, 
    2) get random sequences, 
    3) resize original clips to 512 for gt frames, 
    and 4) resize to 32 for lq frames.

    # NOTE: this is on-the-fly version
    """
    def __init__(self, opt):
        super(TalkingHead1KHTestDataset, self).__init__()

        self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        self.cache_data = opt['cache_data']

        #datapath = 'datasets/TalkingHead-1KH/val/frames'
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']

        self.opt = opt

        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))
        
        ######## my
        subfolders_lq = subfolders_lq[:opt['num_val_clip']]
        subfolders_gt = subfolders_gt[:opt['num_val_clip']]

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial', 'talkinghead1kh']:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                ####### my
                img_paths_lq = img_paths_lq[:opt['num_val_frame_per_clip']]
                img_paths_gt = img_paths_gt[:opt['num_val_frame_per_clip']]

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    print(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        
        # for vfi
        #self.num_gt_frames = opt['num_gt_frames']
        #self.num_lq_frames = self.num_gt_frames // 2 + 1

        self.use_flip = True

        

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

    def imfrombytes(self, content, flag='color', float32=False):
        """Read an image from bytes.
        Args:
            content (bytes): Image bytes got from files or other streams.
            flag (str): Flags specifying the color type of a loaded image,
                candidates are `color`, `grayscale` and `unchanged`.
            float32 (bool): Whether to change to float32., If True, will also norm
                to [0, 1]. Default: False.
        Returns:
            ndarray: Loaded image array.
        """
        img_np = np.frombuffer(content, np.uint8)
        imread_flags = {
            'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
        img = cv2.imdecode(img_np, imread_flags[flag])
        if float32:
            img = img.astype(np.float32) / 255.
        return img

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            #img_gt = self.imgs_gt[folder][idx]
            imgs_gt = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            #img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            #img_gt.squeeze_(0)
            img_paths_gt = [self.imgs_gt[folder][i] for i in select_idx]
            imgs_gt = read_img_seq(img_paths_gt)

        imgs_gt.squeeze_(0), imgs_lq.squeeze_(0)
        if len(lq_path) == 1: 
            lq_path = lq_path[0]
        
        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': imgs_gt, #img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path,  # lq paths # center frame
            'is_lq_given': True,
            'is_gt_given': True
        }

    def __len__(self):
        return len(self.data_info['gt_path'])
        
    '''
    def __getitem__(self, idx):
        video_path = self.videolist[idx]

        video_reader = mmcv.VideoReader(video_path)

        num_frames, fps, h, w = len(video_reader), video_reader.fps, video_reader.height, video_reader.width
        
        random_frame_indices = random.randint(0, num_frames-self.num_gt_frames)
        random_frame_indices = list(
            range(random_frame_indices, random_frame_indices + self.num_gt_frames))
        
        frames = [video_reader[i] for i in random_frame_indices]
        #frames = np.stack(frames, axis=0)

        # bgr. (t h w c) (0~255) (uint8) 
        #frames = frames.transpose(0, 3, 1, 2)
        #frames = (frames / 255).astype(np.float32)

        # numpy to tensor
        #frames = [self.transform(frames[i]) for i in range(frames.shape[0])]
        #frames = np.stack(frames, axis=0)

        # tensor 
        #frames = frames[:, :, :, ::-1] # bgr to rgb
        #frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))

        frames = self.img2tensor(frames, bgr2rgb=True, float32=True)
        frames = torch.stack(frames)

        if not self.use_flip and random.random() < 0.5:
            frames = frames[..., ::-1]

        frames = torch.clamp(frames, 0, 255) / 255

        frames_gt = torch.clamp(resize(frames, out_shape=(512, 512)), 0, 1)
        frames_lq = torch.clamp(resize(frames_gt, out_shape=(32, 32)), 0, 1)

        # normalize
        normalize(frames_gt, self.mean, self.std, inplace=True)
        normalize(frames_lq, self.mean, self.std, inplace=True)
        
        # sparse sampling for vfi
        indices = torch.tensor([i*2 for i in range(self.num_gt_frames//2 + 1)])
        frames_lq = torch.index_select(frames_lq, 0, indices)

        if self.num_gt_frames == 1:
            #print('here')
            #print(frames_gt.shape)
            frames_gt, frames_lq = frames_gt.squeeze(0), frames_lq.squeeze(0)
            #print(frames_gt.shape)

        return {
            'gt': frames_gt, 
            'lq': frames_lq, 
            'gt_path': video_path
        }

    def __len__(self):
        return len(self.videolist)
    '''