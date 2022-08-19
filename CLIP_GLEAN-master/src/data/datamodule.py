from importlib import import_module

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .ffhq_dataset import FFHQDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .celebahq_dataset import CelebAHQDataset
from .folder_dataset import FolderDataset
from .folder_vfi_dataset import FolderVFIDataset
from .talkinghead1kh_dataset import TalkingHead1KHDataset
from .talkinghead1kh_test_dataset import TalkingHead1KHTestDataset, generate_talkinghead_datasets
from .repeat_dataset import RepeatDataset
from .clip_sr_annotation_dataset import CLIPSRAnnotationDataset
from .clip_sr_annotation_iter_dataset import CLIPSRIterAnnotationDataset

benchmark = ['CelebAHQ']


class SRDataModule(LightningDataModule):
    def __init__(self, opt, is_train=False):
        super().__init__()
        self.opt = opt
        self.is_train = is_train

        if is_train:
            self.trainset_name = opt['train']['name']
            self.opt['train']['root_path'] = opt['root_path'] # TODO: need cleanup
            self.opt['train']['caption_path'] = opt['caption_path']

        self.testset_name = opt['test']['name']
        self.opt['test']['root_path'] = opt['root_path'] # TODO: need cleanup
        self.opt['test']['caption_path'] = opt['caption_path']


    def setup(self, stage=None):
        # generate train loader
        if self.is_train:
            if self.trainset_name == 'FFHQ':
                self.train_dataset = FFHQDataset(self.opt['train'])
            elif self.trainset_name == 'FFHQDegradation':
                self.train_dataset = FFHQDegradationDataset(self.opt['train'])
            elif self.trainset_name == 'TalkingHead1KH':
                self.train_dataset = TalkingHead1KHDataset(self.opt['train'])
            # more correct dataset name
            elif self.trainset_name in ['ClipGlean', 'ClipGpen']: 
                self.train_dataset = CLIPSRAnnotationDataset(self.opt['train'])
            elif self.trainset_name in ['ClipGleanIter', 'ClipGpenIter']:
                self.train_dataset = CLIPSRIterAnnotationDataset(self.opt['train'])
            else:
                raise NotImplementedError(f'train dataset not implemented: {self.trainset_name}')
            
            # repeat dataset
            if self.opt['train']['repeat'] is not None:
                self.train_dataset = RepeatDataset(
                    self.train_dataset, 
                    self.opt['train']['repeat'], 
                    iterations=None, #self.opt['train']['iterations'], 
                    batch_size=-1) #self.opt['train']['batch_size'])
        
        # generate test loader
        if self.testset_name == 'CelebAHQ':
            self.test_dataset = CelebAHQDataset(self.opt['test'])
        elif self.testset_name == 'FolderDataset':
            self.test_dataset = FolderDataset(self.opt['test'])
        elif self.testset_name == 'FolderVFIDataset':
            self.test_dataset = FolderVFIDataset(self.opt['test'])
        elif self.testset_name == 'TalkingHead1KH':
            self.test_dataset = generate_talkinghead_datasets(self.opt['test'])
        elif self.testset_name in  ['ClipGlean', 'ClipGpen']: 
            self.test_dataset = CLIPSRAnnotationDataset(self.opt['test'], test=True)
        elif self.testset_name in ['ClipGleanIter', 'ClipGpenIter']: 
                self.test_dataset = CLIPSRIterAnnotationDataset(self.opt['test'], test=True)
        else:
            raise NotImplementedError(f'test dataset not implemented: {self.testset_name}')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.opt['train']['batch_size_per_gpu'], 
            num_workers=self.opt['num_workers'], 
            pin_memory=self.opt['train']['pin_memory'],
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        if self.testset_name  == 'TalkingHead1KH':
            return [DataLoader(
                testset, batch_size=1, num_workers=self.opt['num_workers'], shuffle=False) 
                for testset in self.test_dataset]
        else:
            return DataLoader(
                self.test_dataset, batch_size=1, num_workers=self.opt['num_workers'], shuffle=False)

    def test_dataloader(self):
        if self.testset_name  == 'TalkingHead1KH':
            return [DataLoader(
                testset, batch_size=1, num_workers=self.opt['num_workers'], shuffle=False) 
                for testset in self.test_dataset]
        else:
            return DataLoader(
                self.test_dataset, batch_size=1, num_workers=self.opt['num_workers'], shuffle=False)