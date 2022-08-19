# Copyright (c) OpenMMLab. All rights reserved.


# from ..builder import build_backbone, build_component, build_loss
# from ..common import set_requires_grad
# from ..registry import MODELS
# from .basic_restorer import BasicRestorer

from src.losses.clip_loss import CLIPLoss, MeasureAugCLIP

import os.path as osp
import os

import cv2
import lpips
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule

from src.losses import L1Loss, MSELoss, PerceptualLoss, GANLoss, IDLoss

from src.utils import psnr, tensor2img
from collections import OrderedDict
import clip
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from src.evaluation.metrics import psnr, ssim
import numpy as np
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from torchvision.utils import make_grid
from torch.nn import functional as F

class LitClipGleanIter(LightningModule):
    def __init__(self,
                 net_g: torch.nn.Module,
                 net_d: torch.nn.Module,
                 opt: dict,
                 pretrained=None,
                 strict_load=True):

        super().__init__()

        self.save_hyperparameters(
            ignore=['net_g', 'net_d', 'is_train'])

        self.opt = opt

        self.generator = net_g
        self.discriminator = net_d

        self.set_requires_grad(self.generator, False)
        self.set_requires_grad(self.discriminator, False)
        self.generator.delta_image.requires_grad  = True
        self.generator.delta_code.requires_grad  = True

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        # freeze clip model
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        
        # loss
        self._create_loss(opt['loss'])

        #metric
        self.metric_lpips = lpips.LPIPS(net='vgg')

        self.sampled_images = []
        self.sampled_captions = [] 

    
    def _create_loss(self, opt):
        
        self.loss_gan = GANLoss(
            gan_type='vanilla',
            loss_weight=opt['gan_weight'],
            real_label_val=1.0,
            fake_label_val=0
        ) if opt['gan_weight'] > 0 else None

        self.loss_clip = MeasureAugCLIP(
            self.clip_model,
            loss_weight=opt['clip_weight']
        ) if opt['clip_weight'] > 0 else None 

        self.id_loss = IDLoss(
            opt,
            loss_weight=opt['id_weight']
        ) if opt['id_weight'] > 0 else None
        self.set_requires_grad(self.id_loss, False)

        self.latent_l2_loss = MSELoss(
            loss_weight=opt['latent_l2_weight'],
            reduction='mean'
        ) if opt['latent_l2_weight'] > 0 else None

        self.lr_cons_loss = L1Loss(
            loss_weight=opt['lr_cons_weight'],
            reduction='mean'
        ) if opt['lr_cons_weight'] > 0 else None

    def forward(self, batch):
        lq, caption = batch['lq'], batch['cap']
        with torch.no_grad():
            tokenized_caption = torch.cat([clip.tokenize(cap) for cap in caption]).cuda()
            text_features = self.clip_model.encode_text(tokenized_caption)
            output = self.generator(lq, text_features)
        return output

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all the networks.
        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx):
        """Train step.
        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """

        # data
        lq, gt = batch['lq'], batch['gt']
        caption = batch['cap']

        losses = dict()

        tokenized_caption = torch.cat([clip.tokenize(cap) for cap in caption]).cuda()
        text_features = self.clip_model.encode_text(tokenized_caption)
        fake_g_output,w,w_hat = self.generator(lq, strategy=['image_space', 'code_space'])

        if self.id_loss or self.latent_l2_loss:
            glean_g_output, w_glean, w_hat_glean = self.generator(lq, strategy=[])

        # clip loss
        if self.loss_clip:
            loss_clip = self.loss_clip(fake_g_output, tokenized_caption)
            losses['loss_clip'] = loss_clip.mean() if loss_clip is not None else torch.tensor(.0)

        # lr loss
        if self.lr_cons_loss:
            # random_noise = ((torch.randn_like(lq) - 0.5) * 2) / 255.
            down_fake_g_output = F.interpolate(fake_g_output,size=(lq.shape[2],lq.shape[3]),mode='bicubic', antialias=True)
            losses['loss_lr_cons'] = self.lr_cons_loss(down_fake_g_output,lq)

        # id loss
        if self.id_loss:
            loss_id, sim_improvement = self.id_loss(fake_g_output, glean_g_output)
            losses['loss_id'] = loss_id

        # latent l2 loss
        if self.latent_l2_loss:
            latent_l2 = self.latent_l2_loss(w, w_glean)
            losses['loss_latent_l2'] = latent_l2
        
        # gan loss
        if self.loss_gan:
            fake_g_pred = self.discriminator(fake_g_output)
            losses['loss_gan'] = self.loss_gan(fake_g_pred, target_is_real=True, is_disc=False)

        log_train = {f"train/loss_g/{k.replace('loss_','')}" : v for k,v in losses.items()}
        self.log_dict(log_train, logger=True, prog_bar=True)

        total_loss = sum(losses.values())
        return total_loss


    # TODO Implement validation step
    def validation_step(self, batch, batch_idx):
        lq, gt, caption = batch['lq'], batch['gt'], batch['cap']
        assert len(lq) == 1, 'only support 1 batch size'
        tokenized_caption = torch.cat([clip.tokenize(cap) for cap in caption]).cuda()
        text_features = self.clip_model.encode_text(tokenized_caption)

        # generator - format(-1 ~ +1)
        fake_g_output,_,_ = self.generator(lq, strategy=[])
        output = torch.clamp(fake_g_output, -1, 1)

        test_img,_,_ = self.generator(lq, strategy=['image_space', 'code_space'])
        test_output = torch.clamp(test_img, -1, 1)
        test_output = test_output/2 + 0.5

        # lpips (CHW) - image should be RGB, IMPORTANT: normalized to [-1,1]
        metric_lpips = self.metric_lpips(gt, output)
        
        # change format (0 ~ +1)
        gt = gt/2 + 0.5
        output = output/2 + 0.5
        lq = lq/2 + 0.5
        
        # psnr, ssim (HWC)
        output = output[0].cpu().numpy().transpose(1,2,0)
        gt = gt[0].cpu().numpy().transpose(1,2,0)
        # Images with range [0, 255]
        metric_psnr = psnr(output*255, gt*255) 
        metric_ssim = ssim(output*255, gt*255)

        self.log_dict({
            'val/metric/psnr': metric_psnr,
            'val/metric/ssim': metric_ssim,
            'val/metric/lpips_vgg': metric_lpips,
        })

        if batch_idx < 10:
            self.sampled_images.append(torch.from_numpy(gt.transpose(2,0,1)))
            self.sampled_images.append(torch.from_numpy(output.transpose(2,0,1)))
            self.sampled_images.append(test_output[0].cpu())
            self.sampled_images.append(T.Resize(256, interpolation=InterpolationMode.NEAREST)(lq[0].cpu()))
            self.sampled_images.append(T.Resize(256, interpolation=InterpolationMode.NEAREST)( T.Resize(16, interpolation=InterpolationMode.BICUBIC, antialias=True)(test_output[0].cpu()) ))
            self.sampled_captions.append(caption[0])
        
    # TODO Implement test step
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        grid = make_grid(self.sampled_images, nrow=5)

        
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'val/visualization',
                grid, self.global_step+1, dataformats='CHW',)
            concated_captions = '   '.join([f'[{idx_cap}] ::: {cap}'for idx_cap, cap in enumerate(self.sampled_captions)])
            self.logger.experiment.add_text('caption',  concated_captions, global_step=self.global_step)
        
        self.sampled_images = []
        self.sampled_captions = []


    def test_epoch_end(self, outputs):
        pass

    # TODO check it has same confs comparing original one
    def configure_optimizers(self):
        optimizer_g = Adam(
            self.generator.parameters(), 
            lr=self.opt['optim_g']['lr'],
            betas=self.opt['optim_g']['betas']
        )
    
        scheduler_g = {
            'scheduler': CosineAnnealingLR(
                optimizer_g, 
                T_max=self.opt['scheduler']['T_max'], 
                eta_min=self.opt['scheduler']['eta_min']),
            'name': 'learning_rate_g'
        }

        return [optimizer_g], [scheduler_g]

