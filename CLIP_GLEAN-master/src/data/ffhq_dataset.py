import glob
import random
from os import path as osp

from PIL import Image
from torch.utils import data as data
from torchvision import transforms as T






class FFHQDataset(data.Dataset):
    """FFHQ dataset """
    def __init__(self, opt):
        super(FFHQDataset, self).__init__()
        self.opt = opt

        self.scale = opt['scale']

        self.gt_size = opt['gt_size']
        self.lq_size = opt['gt_size'] // opt['scale']
        self.gt_folder = osp.join(opt['root_path'], 'FFHQ', f'images{self.gt_size}x{self.gt_size}')
        self.gt_paths = sorted(glob.glob(osp.join(self.gt_folder, f'*.png')))
        if self.gt_size == 512:
            self.lq_folder = osp.join(opt['root_path'], 'FFHQ', f'images{self.lq_size}x{self.lq_size}_from512')
        else:
            self.lq_folder = osp.join(opt['root_path'], 'FFHQ', f'images{self.lq_size}x{self.lq_size}')
        self.lq_paths = sorted(glob.glob(osp.join(self.lq_folder, f'*.png')))
        
        self.flip = opt['use_hflip']

        assert len(self.gt_paths) == len(self.lq_paths) and len(self.gt_paths) == 70000, print(f'something is wrong for FFHQ dataset, FFHQ dataset includes 70000 images however we got {len(self.lq_paths)} for lq and {len(self.gt_paths)} for hq')

        self.transform_train = T.Compose([
            T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        lq_path, gt_path = self.lq_paths[idx], self.gt_paths[idx]
        
        img_lq, img_gt = Image.open(lq_path), Image.open(gt_path)
        
        img_lq, img_gt = self.transform_train(img_lq), self.transform_train(img_gt)
        
        if not self.flip and random.random() < 0.5:
            img_lq, img_gt = img_lq[:, :, ::-1], img_gt[:, :, ::-1]

        return {'lq':img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        
    def __len__(self):
        return len(self.gt_paths)




if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/jinsuyoo/datasets')
    parser.add_argument('--data_train', type=str, default='FFHQ')
    parser.add_argument('--data_test', type=str, default='CelebAHQ')

    args = parser.parse_args()

    dataset = FFHQDataset(args)
    dataloader = data.DataLoader(dataset, batch_size=1)

    for batch in dataloader:
        lq, gt, path = batch['lq'], batch['gt'], batch['gt_path']
        print(gt.shape, gt.max(), gt.min(), lq.shape, path)
        input()


    

    