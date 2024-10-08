import config
import tarfile
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


class Dataset:
    def download_dataset(self):
        download_url(url=config.DATASET_URL, root='.', filename='cifar10.tgz', md5='')
        with tarfile.open('cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./input')
        print('[INFO] dataset download completed')

    def get_dataloader(self):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tfms = tt.Compose([tt.RandomCrop(32, padding=4),
                                 tt.RandomHorizontalFlip(),
                                 tt.ToTensor(),
                                 tt.Normalize(*stats)])
        valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
        train_ds = ImageFolder(config.DATA_DIR + '/cifar10/train', train_tfms)
        valid_ds = ImageFolder(config.DATA_DIR + '/cifar10/test', valid_tfms)
        train_dl = DataLoader(train_ds, config.TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
        valid_dl = DataLoader(valid_ds, config.VALID_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
        return train_dl, valid_dl
