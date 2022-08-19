# Copyright (c) OpenMMLab. All rights reserved.
from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss, PerceptualVGG  
from .pixelwise_loss import L1Loss, MSELoss
from .clip_loss import CLIPLoss
from .id_loss import IDLoss

__all__ = ['L1Loss', 'MSELoss', 'GANLoss', 'PerceptualLoss', 'PerceptualVGG', 'clip_loss', 'id_loss']