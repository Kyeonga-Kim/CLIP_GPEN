import os
import os.path as osp
import glob
from tqdm import tqdm
import mmcv

video_paths = sorted(glob.glob('/data/jinsuyoo/TalkingHead-1KH/train/cropped_clips/*.mp4'))

print(len(video_paths))

pbar = tqdm(video_paths)

for i, video_path in enumerate(pbar):
    video_reader = mmcv.VideoReader(video_path)
    num_frames, fps, h, w = len(video_reader), video_reader.fps, video_reader.height, video_reader.width
    #print(num_frames, fps, h, w)
    if not (num_frames > 9): #and h >= 512 and w >= 512):
        os.system(f'mv {video_path} /data/jinsuyoo/TalkingHead-1KH/train/cropped_clips_noframe-or-h-or-w-is-less-than-512')
    # TODO: cropped_clips_lessthan10frame-or-h-or-w-is-less-than-512
    