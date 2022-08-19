from os import path as osp
import glob
from torch.utils import data as data
from torchvision import transforms as T

from PIL import Image


class FolderDataset(data.Dataset):
    def __init__(self, opt):
        super(FolderDataset, self).__init__()
        self.opt = opt

        self.folder_lq = opt['folder_lq']
        self.folder_gt = opt['folder_gt']

        # specifying given data type
        self.is_gt_given = self.folder_gt is not None
        self.is_lq_given = self.folder_lq is not None

        assert self.is_gt_given or self.is_lq_given, 'data folder should be given'
        
        # NOTE: online resizing gt not implemented yet

        if self.is_gt_given:
            self.gt_paths = sorted(glob.glob(osp.join(self.folder_gt, '*')))

        if self.is_lq_given:
            self.lq_paths = sorted(glob.glob(osp.join(self.folder_lq, '*')))
            
        self.transform = T.Compose([
            T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        print(len(self.lq_paths))

    def __getitem__(self, idx):
        if self.is_gt_given:
            gt_path = self.gt_paths[idx]
            img_gt = Image.open(gt_path)
            img_gt = self.transform(img_gt)
        else:
            gt_path, img_gt = 0 ,0 #TODO: cleanup

        if self.is_lq_given:
            lq_path = self.lq_paths[idx]
            img_lq = Image.open(lq_path)
            img_lq = self.transform(img_lq)
        else:
            lq_path, img_lq = 0,0

        return {
            'lq':img_lq, 
            'gt': img_gt, 
            'lq_path': lq_path, 
            'gt_path': gt_path,
            'is_lq_given': self.is_lq_given,
            'is_gt_given': self.is_gt_given}

    def __len__(self):
        return len(self.lq_paths)
