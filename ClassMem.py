import scipy.io
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from tqdm import tqdm
from model import ResNet9


data_in = h5py.File('./IDclass.mat', 'r')
b = np.array(data_in['ans'],dtype='int').reshape(100)
datapath = './data'
s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
flip = transforms.RandomHorizontalFlip()
data_transforms = transforms.Compose([ToTensor(), Normalize(0.5, 0.5)])
CIFAR_10_Dataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                                 transform=data_transforms)
sublist = list(range(0, 2, 1))
subset = torch.utils.data.Subset(CIFAR_10_Dataset, sublist)
dataloader = torch.utils.data.DataLoader(subset, 1, shuffle=False, num_workers=2)

model = ResNet9(3, 10)
model.load_state_dict(torch.load('./model_weights.pth'))


new_m = torchvision.models._utils.IntermediateLayerGetter(model,{'layer3_residual2': 'feat1'})

final = []
final1= []


if __name__ == '__main__':
    for img, label in tqdm(iter(dataloader)):
        out = new_m(img)
        for k, v in out.items():
            my = np.mean(v.reshape(256, 4).cpu().detach().numpy(), axis=1)
            final.append(my)
        out1 = np.mean(np.array(final), axis=0)
        final1.append(out1)
    finalout = np.array(final1)
    maxout = np.max(finalout, axis=0)
    medianout = np.median(np.sort(finalout, axis=0)[0:-1], axis=0)
    selectivity = (maxout - medianout)/(maxout + medianout)
    scipy.io.savemat('./data/selectivity_class.mat', {'selectivity': selectivity})
