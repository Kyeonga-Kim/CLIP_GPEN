import argparse
import os
import os.path as osp
import glob

import mmcv
import numpy as np
import sys

sys.path.append('/home/jinsuyoo/ResizeRight')
from resize_right import resize
from PIL import Image
from tqdm import tqdm

# M009 ...
# audio video
# down front left_30 left_60 right_30 right_60 top
# angry contempt disgusted fear happy neutral sad surprised
# level_1 level_2 level_3
# 001.mp4 002.mp4 ... 029.mp4

'''
python gen_lq_mead.py \
 '/data/jinsuyoo/datasets/MEAD/M009/video/front/happy/level_1/001.mp4' \
 '/home/jinsuyoo/mmediting/jinsu/mead/test'
'''

def get_video_paths(data_path):
    vps = sorted(
        glob.glob(osp.join(data_path, 'MEAD', 'M009', 'video', '*', '*', '*') + '/*.mp4'))
    return vps 



def extract_frames(video_path, save_path, pid, angle, emo, lvl, gt_size=1024, scale_factor=16):
    filename, extension = osp.splitext(osp.basename(video_path))

    os.makedirs(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'hq_raw'), exist_ok=True)
    os.makedirs(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'hq'), exist_ok=True)
    os.makedirs(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'lq'), exist_ok=True)

    video_reader = mmcv.VideoReader(video_path)

    for i, frame in enumerate(video_reader):
        hq_raw = np.flip(frame, axis=2) / 255
        h, w, _ = hq_raw.shape
        hq = hq_raw[(h-gt_size)//2:(h-gt_size)//2 + gt_size, (w-gt_size)//2:(w-gt_size)//2 + gt_size, :]
        
        lq = np.clip(resize(hq, scale_factors=1/scale_factor), 0, 1)

        hq_raw = Image.fromarray((hq_raw * 255).astype(np.uint8))
        hq = Image.fromarray((hq * 255).astype(np.uint8))
        lq = Image.fromarray((lq * 255).astype(np.uint8))

        
        hq_raw.save(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'hq_raw', f'{i:03}.png'))
        hq.save(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'hq', f'{i:03}.png'))
        lq.save(osp.join(save_path, 'MEAD', pid, 'frames', angle, emo, lvl, filename, 'lq', f'{i:03}.png'))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('--save_path', type=str, default='/data/jinsuyoo/datasets')
    parser.add_argument('--data_path', type=str, default='/data/jinsuyoo/datasets')
    args = parser.parse_args()

    video_paths = get_video_paths(data_path=args.data_path)
    for vp in tqdm(video_paths):
        pid, _, angle, emo, lvl, filename = osp.normpath(vp).split(osp.sep)[-6:]

        if lvl == 'level_3' and filename == '001.mp4':
            extract_frames(vp, args.save_path, pid, angle, emo, lvl)