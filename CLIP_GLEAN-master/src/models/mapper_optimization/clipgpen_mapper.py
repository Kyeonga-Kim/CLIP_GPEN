'''
REFERENCE
https://github.com/orpatashnik/StyleCLIP/blob/main/mapper/latent_mappers.py
'''

from torch import nn
from torch.nn import Module
from torch.nn import functional as F
import math
import torch

from .clipgpen_module import ConvLayer, PixelNorm

# STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

class _EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = _fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

def _fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    input = input.cuda()
    if input.ndim == 3:
        return (
            F.leaky_relu(
                input + bias.view(1, *rest_dim, bias.shape[0]), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

class LRMapper(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()

        channels = {
            4: 3,
            8: 4,
            16: 8,
            32: 16,
            64: 32,
            128: 64,
            256: 128,
            512: 256,
        }

        convs = [ConvLayer((512+3), channels[size], 1, device=device)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ConvLayer(in_channel, out_channel, 3, device=device))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = nn.Sequential(
            ConvLayer(3+1, 3, 1, device=device),
            ConvLayer(3, 3, 1, device=device),
            ConvLayer(3, 3, 1, activate=False, device=device),
        )

        # self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        # self.final_linear = nn.Sequential(
        #     _EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
        #     _EqualLinear(channels[4], 1),
        # )

    def forward(self, input, caption=None):
        if caption is not None:
            _, _, h, w = input.shape
            out = torch.cat([input, caption.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w)], dim=1)        
            out = self.convs(out)

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
        else:
            out = torch.zeros_like(input)
        return out


class StyleMapper(Module):
    def __init__(self, device, lr_dim=3, latent_dim=512):
        super(StyleMapper, self).__init__()
        self.device = device

        lr_convs = [
            ConvLayer(lr_dim, lr_dim*(2**2), 3, downsample=True, device=device),
            ConvLayer(lr_dim*(2**2), lr_dim*(4**2), 3, downsample=True, device=device),
            ConvLayer(lr_dim*(4**2), lr_dim*(8**2), 3, downsample=True, device=device),
            ConvLayer(lr_dim*(8**2), 4, 3, downsample=True, device=device), # B C 32 32 
        ]
        lr_linears = [
            _EqualLinear(1024, 512, activation='fused_lrelu'),
            _EqualLinear(512, 512, activation='fused_lrelu'),
        ]
        self.lr_convs = nn.Sequential(*lr_convs)
        self.lr_linears = nn.Sequential(*lr_linears)

        layers = [
            # PixelNorm(),
            _EqualLinear((512+512+512), latent_dim, lr_mul=0.01, activation='fused_lrelu'),
        ]

        for _ in range(3):
            layers.append(
                _EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu', 
                )
            )

        self.mapping = nn.Sequential(*layers)

    def forward(self, lr_plus_delta, style_code, caption=None):
        if caption is not None:
            lr_batch = lr_plus_delta.shape[0]
            lr = self.lr_convs(lr_plus_delta)
            lr = lr.view(lr_batch, -1)
            lr = self.lr_linears(lr)

            lr = lr.unsqueeze(1).repeat((1, style_code.shape[1], 1))
            caption = caption.unsqueeze(1).repeat((1, style_code.shape[1], 1))
            
            x = torch.cat([lr, style_code, caption], dim=2)
            x = self.mapping(x)
        else:
            b = lr_plus_delta.shape[0]
            x = torch.zeros((b,14,512)).to(self.device)
        return x

class MultiLevelStyleMapper(Module):

    def __init__(self, device):
        super(MultiLevelStyleMapper, self).__init__()
        self.device = device

        self.course_mapping = StyleMapper(device, lr_dim=3, latent_dim=512)
        self.medium_mapping = StyleMapper(device, lr_dim=3, latent_dim=512)
        self.fine_mapping = StyleMapper(device, lr_dim=3, latent_dim=512)

    def forward(self, lr_plus_delta, style_code, caption=None):
        if caption is not None:
            style_code_coarse = style_code[:, :4, :]
            style_code_medium = style_code[:, 4:8, :]
            style_code_fine = style_code[:, 8:, :]

            style_code_coarse = self.course_mapping(lr_plus_delta, style_code_coarse, caption)
            style_code_medium = self.medium_mapping(lr_plus_delta, style_code_medium, caption)
            style_code_fine = self.fine_mapping(lr_plus_delta, style_code_fine, caption)

            out = torch.cat([style_code_coarse, style_code_medium, style_code_fine], dim=1)
        else:
            b = lr_plus_delta.shape[0]
            out = torch.zeros((b,14,512)).to(self.device)
        return out

        
class ZeroLRMapper(Module):
    def __init__(self, device):
        super(ZeroLRMapper, self).__init__()
        self.device = device

    def forward(self, input, caption=None):
        return torch.zeros_like(input)

class ZeroStyleMapper(Module):
    def __init__(self, device):
        super(ZeroStyleMapper, self).__init__()
        self.device = device

    def forward(self, lr_plus_delta, style_code, caption=None):
        b = lr_plus_delta.shape[0]
        return torch.zeros((b,14,512)).to(self.device)