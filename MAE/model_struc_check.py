import os

from torch.utils.data import Dataset

import torch.nn as nn
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from tqdm import tqdm
import numpy as np
import model
import mode2
import model_modify_att as att
import model_modify
from utils import setup_seed

test = att.MAE_Encoder(mask_ratio=0.75)
mymodel = model.MAE_ViT(mask_ratio=0.75)
encoder = mymodel.encoder
decoder = mymodel.decoder
encoder.shuffle = model.PatchShuffle(0.75)
test.shuffle = att.PatchShuffle(0.75)
datapath = 'E:/CISPA/Projects/PJ1/MAE_attack/data'
CIFAR_10_train = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                              transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
N=3
sublist = list(range(0, 3, 1))
subset = torch.utils.data.Subset(CIFAR_10_train, sublist)
train_dataloader = torch.utils.data.DataLoader(dataset=subset, batch_size=1,
                                               shuffle=False, num_workers=2)
test.load_state_dict(torch.load('./mae_encoder.pth'), strict=False)

for name,param in encoder.named_parameters():
    print(name, param.shape)
mylist = []
print('-------------------------------')
for name,param in test.named_parameters():
    print(name, param.shape)
mylist = []

'''
if __name__ == '__main__':
    for img, label in tqdm(iter(train_dataloader)):
        out, i = test(img)

        mylist.append(out.detach().numpy())
    output = np.array(mylist).reshape(N, 65, 192)
    print(output)
'''

