# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint
from torch.nn import functional as F

from mmedit.utils import get_root_logger
# from ..registry import LOSSES

# import clip

# @LOSSES.register_module()
class CLIPLoss(nn.Module):
    def __init__(self, model, loss_weight):
        super(CLIPLoss, self).__init__()
        self.model = model
        self.loss_weight = loss_weight

    # TODO check upsample method
    def forward(self, image, text):
        if self.loss_weight <= 0.0: return None
        resized_image = F.interpolate(image, size=(224, 224), mode='bicubic', antialias=True)
        similarity = 1 - self.model(resized_image, text)[0] / 100
        return similarity

# TODO check new version!
# class CLIPLoss(nn.Module):
#     def __init__(self):
#         super(CLIPLoss, self).__init__()

#     def forward(self, image_features, text_features):
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         similarity = image_features @ text_features.T
#         return 1 - similarity

# https://github.com/gnobitab/FuseDream/blob/3fe28b8db3e075c358ea8030b157007ad056ae74/fusedream_utils.py

from src.losses.DiffAugment_pytorch import DiffAugment

class MeasureAugCLIP(nn.Module):
    def __init__(self,clip_model,loss_weight):
        super().__init__()
        self.LATENT_NOISE = 0.01
        self.Z_THRES = 2.0
        self.POLICY = 'color,translation,resize,cutout'
        self.interp_mode = 'bicubic'
        # self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        # self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        self.clip_model = clip_model
        self.loss_weight = loss_weight
    
    def forward(self, image_tensors, text_token, num_samples=1):
        avg_loss = 0.0
        for itr in range(num_samples):
            for j in range(image_tensors.shape[0]):
                loss = self.augmentLoss(image_tensors[j:(j+1)], self.clip_model, text_token, replicate=50, interp_mode=self.interp_mode)
            avg_loss += loss

        avg_loss /= num_samples
        return avg_loss * self.loss_weight
    
    def augmentLoss(self, img, clip_model, text, replicate=10, interp_mode='bilinear'):
        clip_c = clip_model.logit_scale.exp()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(img.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(img.device)
        img_aug = DiffAugment(img.repeat(replicate, 1, 1, 1), policy=self.POLICY)
        img_aug = (img_aug+1.)/2.
        img_aug = F.interpolate(img_aug, size=224, mode=interp_mode)
        img_aug.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

        logits_per_image, logits_per_text = clip_model(img_aug, text)
        logits_per_image = logits_per_image / clip_c
        concept_loss = (-1.) * logits_per_image 
        
        return concept_loss.mean(dim=0, keepdim=False)
