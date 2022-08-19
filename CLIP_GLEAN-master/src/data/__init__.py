from .datamodule import SRDataModule


def create_datamodule(opt, is_train):
    return SRDataModule(opt, is_train=is_train)