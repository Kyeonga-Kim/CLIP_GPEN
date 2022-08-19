import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from torch.nn import functional as F



class PerceptualVGG(nn.Module):
    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        _vgg = vgg.vgg16(pretrained=True)
        _vgg.eval()
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output


class PerceptualLoss(nn.Module):
    def __init__(self,
                 layer_weights={'21': 1.0},
                 vgg_type='vgg16',
                 use_input_norm=True,
                 perceptual_weight=1e-2,
                 norm_img=False,
                 pretrained='torchvision://vgg16',
                 criterion='mse'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights

        self.vgg = PerceptualVGG(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        return percep_loss


if __name__ == '__main__':
    vgg_loss = PerceptualLoss()
    a = torch.randn(1, 3, 1024, 1024)
    b = torch.randn(1, 3, 1024, 1024)
    loss = vgg_loss(a, b)
    print(loss)