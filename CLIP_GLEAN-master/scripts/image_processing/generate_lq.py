import sys
sys.path.append('./ResizeRight')

from resize_right import resize
import os.path as osp
import glob
import argparse
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from multiprocessing import Pool
import cv2



# python generate_lq.py --gt_path '/data/jinsuyoo/datasets/FFHQ/images1024x1024' --save_path '/data/jinsuyoo/datasets/FFHQ/images64x64' --scale 16 --n_thread 12
# python generate_lq.py --gt_path '/data/jinsuyoo/datasets/1024x1024/data1024x1024' --save_path '/data/jinsuyoo/datasets/1024x1024/data64x64' --scale 16 --n_thread 12

#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/CelebAHQ/images1024x1024_png' --save_path '/data/jinsuyoo/datasets/CelebAHQ/images64x64_final' --scale 16 --n_thread 12

#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/CelebAHQ/images1024x1024' --save_path '/data/jinsuyoo/datasets/CelebAHQ/images32x32' --scale 32 --n_thread 12

#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/CelebAHQ/images1024x1024' --save_path '/data/jinsuyoo/datasets/CelebAHQ/images512x512' --scale 2 --n_thread 12

#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/FFHQ/images1024x1024' --save_path '/data/jinsuyoo/datasets/FFHQ/images32x32' --scale 32 --n_thread 12



#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/FFHQ/images512x512' --save_path '/data/jinsuyoo/datasets/FFHQ/images32x32_from512' --scale 16 --n_thread 12
#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/CelebAHQ/images512x512' --save_path '/data/jinsuyoo/datasets/CelebAHQ/images32x32_from512' --scale 16 --n_thread 12

#python generate_lq.py --gt_path '../../datasets/mead/test/hq512x512' --save_path '../../datasets/mead/test/lq32x32' --scale 16 --n_thread 12

#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/FFHQ/images1024x1024' --save_path '/data/jinsuyoo/datasets/FFHQ/images16x16' --scale 64 --n_thread 12
#python generate_lq.py --gt_path '/data/jinsuyoo/datasets/CelebAHQ/images1024x1024' --save_path '/home/kka0602/dataset/CelebAHQ/images256x256' --scale 4 --n_thread 12

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, help='path to gt image folder')
parser.add_argument('--save_path', type=str, help='path to save')
parser.add_argument('--scale', type=int, help='down-scale factor')
parser.add_argument('--n_thread', type=int, help='number of threads')
args = parser.parse_args()

# NOTE: add file extension
gt_paths = glob.glob(osp.join(args.gt_path, '*.png')) 

def worker(gt_path, args):
    img_name, extension = osp.splitext(osp.basename(gt_path))
    img = np.array(Image.open(gt_path).convert('RGB')) / 255
    h, w, c = img.shape
    img_lr = np.clip(resize(img, out_shape=(h/args.scale, w/args.scale)), 0, 1)
    img_lr = (img_lr * 255).round().astype(np.uint8)
    #img_lr = Image.fromarray((img_lr * 255).round().astype(np.uint8))
    #img_lr.save(osp.join(args.save_path, f'{img_name}{extension}'))
    img_lr = img_lr[..., ::-1]
    cv2.imwrite(osp.join(args.save_path, f'{img_name}{extension}'), img_lr)
    
    process_info = f'Processing {img_name} ...'
    return process_info 

os.makedirs(args.save_path, exist_ok=True)

pbar = tqdm(total=len(gt_paths), unit='image', desc='Downscale')

pool = Pool(args.n_thread)
for gt_path in gt_paths:
    pool.apply_async(worker, args=(gt_path, args), callback=lambda arg: pbar.update(1))
pool.close()
pool.join()
pbar.close()
print('All processes done.')