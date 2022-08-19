'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import math
import random
import functools
import operator
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .clipgpen_module import *
from .clipgpen_mapper import LRMapper, ZeroLRMapper, StyleMapper, MultiLevelStyleMapper, ZeroStyleMapper

class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        mapper={},
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cuda'
    ):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        # Mapper
        mappers = {
            'StyleMapper': StyleMapper,
            'MultiLevelStyleMapper': MultiLevelStyleMapper,
        }
        self.latent_mapper = mappers[mapper['latent']](device=device) if mapper['latent'] in mappers else ZeroStyleMapper(device=device)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device
                )
            )

        self.style = nn.Sequential(*layers)
        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
        )
        self.to_rgb1 = ToRGB(self.channels[4]*self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel*self.feat_multiplier,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    isconcat=isconcat,
                    device=device
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel*self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
                )
            )

            self.to_rgbs.append(ToRGB(out_channel*self.feat_multiplier, style_dim, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
    
    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        lr_plus_delta,
        caption,
        latent_code=None,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        strategy=[]
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            '''
            noise = [None] * (2 * (self.log_size - 2) + 1)
            '''
            noise = []
            batch = styles[0].shape[0]
            for i in range(self.n_mlp + 1):
                size = 2 ** (i+2)
                noise.append(torch.randn(batch, self.channels[size], size, size, device=styles[0].device))
            
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # TODO : Clone하는게 맞나?
        # Latent Code
        latent_code = latent_code if latent_code else self.latent_mapper(lr_plus_delta=lr_plus_delta.clone(), 
                                                                         style_code=latent.clone(), 
                                                                         caption=caption)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0] + latent_code[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1] + latent_code[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i] + latent_code[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1] + latent_code[:,  i + 1], noise=noise2) #latent : (1,14,512)? #latent_code : (1,8,512)
            skip = to_rgb(out, latent[:, i + 2] + latent_code[:,  i + 2] , skip)
            i += 2

        image = skip

        return image, latent, latent_code


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1, device='cpu'):
        super().__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        convs = [ConvLayer(3, channels[size], 1, device=device)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, device=device))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out

class FullGeneratorCLIPMapper(nn.Module):
    def __init__(
        self,
        mapper,
        size=256,
        style_dim=512,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cuda',
        pretrained=None
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        self.log_size = int(math.log(size, 2)) #size = 256
        self.n_latent = self.log_size * 2 - 2

        # LR Mapper
        mappers = {
            'LRMapper': LRMapper,
        }
        self.image16_mapper = mappers[mapper['image16']](size=512, device=device) if mapper['image16'] in mappers else ZeroLRMapper(device=device)

        # Encoder
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))

        # Main Generator
        self.generator = Generator(size, style_dim, n_mlp, mapper, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow, device=device)

        # Pretrained
        if pretrained:
            ckpt_path = pretrained['ckpt_path']
            self.load_state_dict(torch.load(ckpt_path),strict=False)

    def _get_image16_code(self, strategy, image16_code):
        if 'image16' in strategy:
            image16_code = self.mapper()
        return self.lr_mapper() if 'image16' in strategy else torch.zeros_like(self.image16_code)
    
    def _get_image256_mapper(self, strategy):
        return self.image256_code if 'image256' in strategy else torch.zeros_like(self.image256_code)

    def _get_latent_mapper(self, strategy):
        return self.latent_code if 'latent' in strategy else torch.zeros_like(self.latent_code)

    def forward(self,
        inputs,
        caption=None,
        image16_code=None,
        image256_code=None,
        latent_code = None,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        strategy=['image16', 'latent']
    ):
        
        # Strategy Fixed
        image16_code = image16_code if 'image16' in strategy else None
        # image256_code = image256_code if 'image256' in strategy else None
        latent_code = latent_code if 'latent' in strategy else None

        # Code = Mapper(input) = Delta
        image16_code = image16_code if image16_code is not None else self.image16_mapper(inputs, caption)
        # image256_code = image256_code if image256_code is not None else self.image256_mapper(inputs, caption)

        inputs = inputs + image16_code
        inputs = F.interpolate(inputs,size=(256,256),mode='bicubic', antialias=True)
        # inputs = inputs + image256_code
        lr_plus_delta = inputs.clone()
            
        # Encoder
        noise = []
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        outs = self.final_linear(inputs)
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
    
        # Generatorstyles,
        output, latent, latent_code = self.generator([outs], 
                                                    lr_plus_delta,
                                                    caption,
                                                    latent_code, 
                                                    inject_index, truncation, truncation_latent, input_is_latent, noise=noise[1:],
                                                    strategy=strategy)

        return output, latent, image16_code, latent_code
        