import torch

from importlib import import_module

from .clipglean_iter_module import LitClipGleanIter
from .clipgpen_iter_module import LitClipGpenIter
import mmcv

def create_model(opt, is_train=True):

    model_type = opt['type'].lower()

    if is_train:
        if model_type == 'clipgleaniter':
            from .clipglean_iter_styleganv2_net import CLIPGLEANStyleGANv2 as model
            from src.models.mmgen.models.architectures.stylegan import StyleGAN2Discriminator

            generator = model(**opt['generator'])
            discriminator = StyleGAN2Discriminator(**opt['discriminator'])
            
            litmodule = LitClipGleanIter(net_g=generator, net_d=discriminator, opt=opt)
            
        elif model_type ==  'clipgpen':
            from .clipgpen_iter_net import  FullGeneratorCLIP

            #gpen g
            generator = FullGeneratorCLIP(256, channel_multiplier=1, narrow=0.5)
            generator.load_state_dict(torch.load('/root/nas_dajinhan/models/GPEN/GPEN-BFR-256.pth'),strict=False)

            discriminator_type = opt['discriminator']['type']
            if discriminator_type == 'StyleGANv2Discriminator':
                from src.models.mmgen.models.architectures.stylegan import StyleGAN2Discriminator
                discriminator = StyleGAN2Discriminator(**opt['discriminator']['args'])
            elif discriminator_type == 'GpenDiscriminator':
                from .clipgpen_iter_net import  Discriminator
                discriminator = Discriminator(256, channel_multiplier=1, narrow=1)
                discriminator.load_state_dict(torch.load('/root/nas_dajinhan/models/GPEN/GPEN-BFR-256-D.pth'))
            elif discriminator_type == 'TediGANDiscriminator':
                from src.models.tedigan.model import Discriminator
                discriminator = Discriminator(1024)
                discriminator.load_state_dict(torch.load('/root/nas_dajinhan/models/TediGAN/stylegan2-ffhq-config-f.pt')['d'], strict=True)
            else:
                raise ValueError(f'Discriminator [{discriminator_type}] is not supported')

            litmodule = LitClipGpenIter(net_g=generator, net_d=discriminator, opt=opt)

        else:
            raise ValueError(f'Model [{model_type}] is not supported')
    else:
        if model_type == 'clipgleaniter':
            from .clipglean_iter_styleganv2_net import CLIPGLEANStyleGANv2 as model
            from src.models.mmgen.models.architectures.stylegan import StyleGAN2Discriminator

            generator = model(**opt['generator'])
            discriminator = StyleGAN2Discriminator(**opt['discriminator'])

            if opt['ckpt_path']:
                print(f"load checkpoint from {opt['ckpt_path']}")
                litmodule = LitClipGleanIter.load_from_checkpoint(
                    checkpoint_path=opt['ckpt_path'], 
                    strict=False, # debug
                    net_g=generator, 
                    net_d=discriminator,
                    opt=opt,
                    is_train=False
                )
            else:
                litmodule = LitClipGleanIter(net_g=generator, net_d=discriminator, opt=opt)
                
        elif model_type ==  'clipgpen':
            from .clipgpen_iter_net import FullGenerator_SR, Discriminator

            generator = FullGeneratorCLIP(256, channel_multiplier=1, narrow=0.5)
            generator.load_state_dict(torch.load('/root/nas_dajinhan/models/GPEN/GPEN-BFR-256.pth'),strict=False)

            discriminator_type = opt['discriminator']['type']
            if discriminator_type == 'StyleGANv2Discriminator':
                from src.models.mmgen.models.architectures.stylegan import StyleGAN2Discriminator
                discriminator = StyleGAN2Discriminator(**opt['discriminator']['args'])
            elif discriminator_type == 'GpenDiscriminator':
                from .clipgpen_iter_net import  Discriminator
                discriminator = Discriminator(256, channel_multiplier=1, narrow=1)
                discriminator.load_state_dict(torch.load('/root/nas_dajinhan/models/GPEN/GPEN-BFR-256-D.pth'))
            elif discriminator_type == 'TediGANDiscriminator':
                from src.models.tedigan.model import Discriminator
                discriminator = Discriminator(256, channel_multiplier=1, narrow=1)
                discriminator.load_state_dict(torch.load('/root/nas_dajinhan/models/TediGAN/stylegan2-ffhq-config-f.pt'))
            else:
                raise ValueError(f'Discriminator [{discriminator_type}] is not supported')

            if opt['ckpt_path']:
                print(f"load checkpoint from {opt['ckpt_path']}")
                litmodule = LitClipGpenIter.load_from_checkpoint(
                    checkpoint_path=opt['ckpt_path'], 
                    strict=False, # debug
                    net_g=generator, 
                    net_d=discriminator,
                    opt=opt,
                    is_train=False
                )
            else:
                litmodule = LitClipGpenIter(net_g=generator, net_d=discriminator, opt=opt)
                

        else:
            raise ValueError(f'Model [{model_type}] is not supported')
        
    return litmodule