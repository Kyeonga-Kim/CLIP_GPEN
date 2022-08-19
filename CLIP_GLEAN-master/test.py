import os
import yaml
import argparse
from os import path as osp

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from src.models import create_model
from src.data import create_datamodule


config_parser = parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', 
                    default = '/home/kka0602/CLIP_GLEAN/configs/test/test_clipglean.yaml', 
                    type    = str, 
                    metavar = 'FILE',
                    help    = 'YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='VGLEAN Test')

# path configuration
parser.add_argument('--seed', type=int, default=310)

args_config, remaining = config_parser.parse_known_args()

if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

# The main arg parser parses the rest of the args, the usual
# defaults will have been overridden if config file specified.
args = parser.parse_args(remaining)


def main():
    # make directory to save experiment
    args.save_path = args.network['save_path'] = osp.join('experiments/test', args.name)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'images'), exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'videos'), exist_ok=True)

    print(f'Experimental results will be saved at: {args.save_path}')

    # fix random seed
    seed_everything(args.seed)
    
    # create model
    # TODO need scale ?
    # args.network['generator']['scale'] = args.scale
    model = create_model(args.network, is_train=False)

    # create datamodule
    args.data['scale'] = args.scale
    args.data['test']['scale'] = args.scale

    datamodule = create_datamodule(args.data, is_train=False)
    
    # define lightning trainer
    trainer = Trainer(
        accelerator='gpu', 
        devices=args.gpus,
        default_root_dir=args.save_path,
    )
    # trainer = Trainer(
    #     accelerator='gpu' if not args.use_cpu else 'cpu',
    #     devices=args.gpus if not args.use_cpu else None,
    #     default_root_dir=args.save_path,
    #     precision=args.precision
    # )

    # add callbacks
    # if args.save_video:
    #     from src.callbacks import VideoGenerator
    #     args.network['save_video'] = True
    #     trainer.callbacks.append(
    #         VideoGenerator(args.data['test']['num_val_clip']))
    #     print('video clip will be saved')

    # else:
    #     args.network['save_video'] = False

    # begin test
    print(f'start test!')
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()