import os, glob
import yaml
import argparse
from os import path as osp
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.models.mapper_optimization import create_model
from src.data import create_datamodule

config_parser = parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', 
                    default = 'configs/train_mapper/clipgpen_mapper.yaml', 
                    type    = str, 
                    metavar = 'FILE',
                    help    = 'YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='CLIP-GPEN Mapper Training')

# path configuration
parser.add_argument('--seed', type=int, default=310)
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint?')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='path to previous checkpoint')
parser.add_argument('--test', action='store_true', default=False,
                    help='test mode')

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
    args.save_path = args.network['save_path'] = osp.join('experiments/train_mapper', args.name)
    if args.log_path:
        args.save_path = osp.join(args.log_path, args.save_path)
    print(args.save_path, flush=True)
    os.makedirs(args.save_path, exist_ok=True)

    print(f'Experimental results will be saved at: {args.save_path}')

    # fix random seed
    seed_everything(args.seed)
    
    # define logger object
    os.makedirs(osp.join(args.save_path, 'tb_logs'), exist_ok=True)
    logger = TensorBoardLogger(osp.join(args.save_path, 'tb_logs'), name='clipglean')

    # create model
    model = create_model(args.network, is_train=True)

    # create datamodule
    args.data['train']['scale'] = args.scale
    args.data['test']['scale'] = args.scale
    args.data['train']['fixed_captions'] = args.data['fixed_captions']
    args.data['test']['fixed_captions'] = args.data['fixed_captions']
    datamodule = create_datamodule(args.data, is_train=True)
        
    # specify checkpoint configs
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(args.save_path, 'checkpoint'),
        filename='best-epoch{epoch:02d}-psnr{val/metric/psnr:.2f}',
        monitor='val/metric/psnr',
        save_last=True,
        save_top_k=0,
        mode='max',
        auto_insert_metric_name=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # define lightning trainer
    trainer = Trainer(
        strategy='ddp',
        accelerator='gpu', 
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.save_path,
        max_epochs=args.epochs, 
        limit_train_batches=args.limit_train_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_val_steps,
        resume_from_checkpoint=args.ckpt_path if args.resume else None
    )
    
    # begin training
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    main()
