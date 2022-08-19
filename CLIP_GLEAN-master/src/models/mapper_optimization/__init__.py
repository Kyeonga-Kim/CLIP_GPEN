import torch

# Generators
def get_clipmapper_generator(args):
    from .clipgpen_net import  FullGeneratorCLIPMapper
    generator = FullGeneratorCLIPMapper(**args) # checkpoint adjusted
    return generator

# Discriminators
def get_gpen_discriminator(args):
    from .clipgpen_net import  Discriminator
    discriminator = Discriminator(256, channel_multiplier=1, narrow=1)
    if args['pretrained'] is not None:
        ckpt_path = args['pretrained']['ckpt_path']
        discriminator.load_state_dict(torch.load(ckpt_path))
    return discriminator
    
def get_styleganv2_discriminator(args):
    from src.models.mmgen.models.architectures.stylegan import StyleGAN2Discriminator
    discriminator = StyleGAN2Discriminator(**args) # checkpoint adjusted
    return discriminator

def get_tedigan_discriminator(args):
    from src.models.tedigan.model import Discriminator
    discriminator = Discriminator(1024)
    if args['pretrained'] is not None:
        ckpt_path = args['pretrained']['ckpt_path']
        prefix = args['pretrained']['prefix']
        discriminator.load_state_dict(torch.load(ckpt_path)[prefix], strict=True)
        # discriminator.load_state_dict(torch.load('/root/nas_dajinhan/models/TediGAN/stylegan2-ffhq-config-f.pt')['d'], strict=True)


def create_model(opt, is_train=True):
    model_type = opt['type'].lower()

    # Train
    if is_train:
        if model_type ==  'clipgpen':
            from .lit_clipgpen import LitClipGpen

            # Generator
            generator_type = opt['generator']['type']
            get_generator = {
                'FullGeneratorCLIPMapper': get_clipmapper_generator}
            try:
                generator = get_generator[generator_type](opt['generator']['args'])
            except:
                raise ValueError(f'Generator [{generator_type}] is not supported')

            # Discriminator
            discriminator_type = opt['discriminator']['type']
            get_discriminator = {
                'StyleGANv2Discriminator': get_styleganv2_discriminator,
                'GpenDiscriminator': get_gpen_discriminator,
                'TediGANDiscriminator': get_tedigan_discriminator}
            try:
                discriminator = get_discriminator[discriminator_type](opt['discriminator']['args'])
            except:
                raise ValueError(f'Discriminator [{discriminator_type}] is not supported')

            # LitModule
            litmodule = LitClipGpen(net_g=generator, net_d=discriminator, opt=opt)

        else:
            raise ValueError(f'Model [{model_type}] is not supported')

    # Test
    else:
        if model_type ==  'clipgpen':
            from .lit_clipgpen import LitClipGpen
            from .clipgpen_net import  FullGeneratorCLIPMapper

            generator_type = opt['generator']['type']
            get_generator = {
                'FullGeneratorCLIPMapper': get_clipmapper_generator}
            try:
                generator = get_generator[generator_type](opt['generator']['args'])
            except:
                raise ValueError(f'Generator [{generator_type}] is not supported')

            # Discriminator
            discriminator_type = opt['discriminator']['type']
            get_discriminator = {
                'StyleGANv2Discriminator': get_styleganv2_discriminator,
                'GpenDiscriminator': get_gpen_discriminator,
                'TediGANDiscriminator': get_tedigan_discriminator}
            try:
                discriminator = get_discriminator[discriminator_type](opt['discriminator']['args'])
            except:
                raise ValueError(f'Discriminator [{discriminator_type}] is not supported')


            # LitModule
            if opt['ckpt_path']:
                print(f"load checkpoint from {opt['ckpt_path']}")
                litmodule = LitClipGpen.load_from_checkpoint(
                    checkpoint_path=opt['ckpt_path'], 
                    strict=False, # debug
                    net_g=generator, 
                    net_d=discriminator,
                    opt=opt,
                    is_train=False)
            else:
                litmodule = LitClipGpen(net_g=generator, net_d=discriminator, opt=opt)                

        else:
            raise ValueError(f'Model [{model_type}] is not supported')
        
    return litmodule