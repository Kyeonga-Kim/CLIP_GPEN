import os.path as osp
from typing import Any, Optional

import cv2
import numpy as np 

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, 
                 every_n_steps_train=1000,
                 every_n_steps_val=5, 
                 nrow=1):
        super().__init__()

        self.every_n_steps_train = every_n_steps_train
        self.every_n_steps_val = every_n_steps_val
        self.nrow = nrow

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        
        if batch_idx % self.every_n_steps_train == 0:
            # inputs are numpy array 0~255 h w c
            # 0 for generator outputs
            img_lq = outputs[0]['img_lq']
            img_gt = outputs[0]['img_gt']
            output = outputs[0]['output']

            img_lq = (img_lq * 0.5 + 0.5).clamp(0, 1)
            img_gt = (img_gt * 0.5 + 0.5).clamp(0, 1)
            output = (output * 0.5 + 0.5).clamp(0, 1)

            if img_gt.ndim == 5:
                # 2, 3
                b, t, c, h, w = img_gt.shape
                h_lq, w_lq = img_lq.shape[-2:]

                num_lq_frames, num_gt_frames = img_lq.shape[1], img_gt.shape[1]
                lq_indices = np.array(list(range(num_lq_frames))) * 2 

                new_img_lq = []
                for i in range(num_gt_frames):
                    if i % 2 == 0:
                        new_img_lq.append(img_lq[:, i//2])
                    else:
                        new_img_lq.append(torch.zeros_like(img_lq[:, i//2]))
                img_lq = torch.stack(new_img_lq, dim=1)

                img_lq = img_lq.view(b * t, c, h_lq, w_lq)
                img_gt = img_gt.view(b * t, c, h, w)
                output = output.view(b * t, c, h, w)
            else:
                t = 1

            grid = torchvision.utils.make_grid(img_lq, nrow=2*t)
            # divided by n gpus # NOTE: possibly incorrect if ngpu is not 2
            trainer.logger.experiment.add_image(
                f'train/img_lq', grid, global_step=trainer.global_step//2)

            grid = torchvision.utils.make_grid(output, nrow=2*t)
            trainer.logger.experiment.add_image(
                f'train/output', grid, global_step=trainer.global_step//2) 

            grid = torchvision.utils.make_grid(img_gt, nrow=2*t)
            trainer.logger.experiment.add_image(
                f'train/img_gt', grid, global_step=trainer.global_step//2)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        if batch_idx % self.every_n_steps_val == 0:
            # inputs are numpy array 0~255 h w c
            img_lq = outputs['img_lq']
            img_gt = outputs['img_gt']
            output = outputs['output']

            if img_gt.ndim == 5:
                # 2, 3
                b, t, c, h, w = img_gt.shape
                h_lq, w_lq = img_lq.shape[-2:]
                num_lq_frames, num_gt_frames = img_lq.shape[1], img_gt.shape[1]
                lq_indices = np.array(list(range(num_lq_frames))) * 2 

                new_img_lq = []
                for i in range(num_gt_frames):
                    if i % 2 == 0:
                        new_img_lq.append(img_lq[:, i//2])
                    else:
                        new_img_lq.append(torch.zeros_like(img_lq[:, i//2]))
                img_lq = torch.stack(new_img_lq, dim=1)

                img_lq = img_lq.view(b * t, c, h_lq, w_lq)
                img_gt = img_gt.view(b * t, c, h, w)
                output = output.view(b * t, c, h, w)
            else:
                t = 1

            grid = torchvision.utils.make_grid(
                output.squeeze(0).clamp(0, 1), nrow=1*t)
            trainer.logger.experiment.add_image(
                f'val/samples/{batch_idx}_output', grid, global_step=trainer.current_epoch)

            grid = torchvision.utils.make_grid(
                img_lq.squeeze(0).clamp(0, 1), nrow=1*t)
            trainer.logger.experiment.add_image(
                f'val/samples/{batch_idx}_lq', grid, global_step=trainer.current_epoch)

            grid = torchvision.utils.make_grid(
                img_gt.squeeze(0).clamp(0, 1), nrow=1*t)
            trainer.logger.experiment.add_image(
                f'val/samples/{batch_idx}_gt', grid, global_step=trainer.current_epoch)


class VideoGenerator(Callback):
    def __init__(self, num_test_clip=2):

        super().__init__()

        self.img_lq = [[]] * num_test_clip
        self.img_gt = [[]] * num_test_clip
        self.output = [[]] * num_test_clip
    '''
    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
                        
        self.img_lq[dataloader_idx].append(outputs['img_lq'])
        self.img_gt[dataloader_idx].append(outputs['img_gt'])
        self.output[dataloader_idx].append(outputs['output'])

        self.is_gt_given = outputs['is_gt_given'] # possible bug
    
    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module, dataloader_idx):
        video_save_path = osp.join(pl_module.opt['save_path'], 'videos')
        
        self._generate_video_from_frames(
            self.img_lq[dataloader_idx], fps=30, save_path=video_save_path, save_name=f'video_lq{dataloader_idx}')
        self._generate_video_from_frames(
            self.output[dataloader_idx], fps=30, save_path=video_save_path, save_name=f'video_output{dataloader_idx}')
        if self.is_gt_given:
            self._generate_video_from_frames(
                self.img_gt[dataloader_idx], fps=30, save_path=video_save_path, save_name=f'video_gt{dataloader_idx}')
    '''
    def _generate_video_from_frames(self, frames, fps, save_path, save_name):
        h, w, c = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(
            osp.join(save_path, f'{save_name}.mp4'), fourcc, fps, (h, w))

        for frame in frames:
            video_writer.write(frame)

        video_writer.release()

        print(f'video clip {save_name} saved!')