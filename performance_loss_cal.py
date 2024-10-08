import torch
import config

from model import ResNet9
import torchvision
from torchvision import models, transforms
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize

from tqdm import tqdm


def cal(inacc):
    outacc = sum(inacc)/len(inacc)
    return outacc
import numpy as np


mode = '20'
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
model = ResNet9(3,10)
model.load_state_dict(torch.load('./data/model_weights.pth'))
model1 = ResNet9(3,10)
if mode == 'high':
    model1.load_state_dict(torch.load('./data/model_weights_ori_zeros_high.pth'))
elif mode == 'rand':
    model1.load_state_dict(torch.load('./data/model_weights_ori_zeros_rand.pth'))
elif mode == 'low':
    model1.load_state_dict(torch.load('./data/model_weights_ori_zeros_low.pth'))
elif mode == '20':
    model1.load_state_dict(torch.load('./data/model_weights_ori_zeros_20.pth'))
datapath = './data'
s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
flip = transforms.RandomHorizontalFlip()
data_transforms = transforms.Compose(
            [
                #transforms.RandomResizedCrop(size=32),
                #transforms.RandomApply([flip], p=0.5),
                #transforms.RandomApply([color_jitter], p=0.9),
                #transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                Normalize(0.5, 0.5)
            ])
traindataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                            transform=data_transforms)

testdataset = torchvision.datasets.CIFAR10(datapath, train=False, download=False,
                                           transform=data_transforms)
sublist = list(range(0, 45000, 1))
subset = torch.utils.data.Subset(traindataset, sublist)

train_dataloader = torch.utils.data.DataLoader(traindataset, config.TRAIN_BATCH_SIZE, shuffle=True,
                                               num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(testdataset, config.VALID_BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True)
test_dataloader1 = torch.utils.data.DataLoader(testdataset, config.VALID_BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True)
device = torch.device('cuda')
model = model.to(device)
model1 = model1.to(device)

if __name__ == '__main__':
    accuracy = []
    accuracy1 = []
    for img, lb in tqdm(iter(test_dataloader)):
        img = img.to(device)
        lb = lb.to(device)
        output = model(img)
        output1 = model1(img)
        acc = acc_fn(output, lb)
        accuracy.append(acc.item())
        acc1 = acc_fn(output1, lb)
        accuracy1.append(acc1.item())
    avg_acc = cal(accuracy)
    avg_acc1 = cal(accuracy1)
    print(f'Avg valid acc: {avg_acc}')
    print(f'Avg valid acc: {avg_acc1}')
    print(f'Avg valid acc down: {avg_acc-avg_acc1}')




