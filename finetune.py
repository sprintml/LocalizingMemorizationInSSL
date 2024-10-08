import torch
import config
from dataset import Dataset
from model import ResNet9
import torchvision
from torchvision import models, transforms
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from engine import train_fn, eval_fn

neuron_index_to_train = 3
s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
flip = transforms.RandomHorizontalFlip()
data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32),
                transforms.RandomApply([flip], p=0.5),
                transforms.RandomApply([color_jitter], p=0.9),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                Normalize(0.5, 0.5)
            ])
model = ResNet9(3, 10)
model.load_state_dict(torch.load('./model_weights0.pth'))
datapath = './data'
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
model.train()
for name, param in model.named_parameters():
    if name.startswith('layer3_residual2.conv') and 'weight' in name:
        param.data[neuron_index_to_train].requires_grad = True
    elif name.startswith('layer3_residual2.conv') and 'bias' in name:
        param.data[neuron_index_to_train].requires_grad = True
    else:
        param.requires_grad = False

device = torch.device('cuda')
model = model.to(device)
steps = len(train_dataloader) * config.EPOCHS
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.LEARNING_RATE, epochs=config.EPOCHS,
                                                total_steps=steps)

if __name__ == '__main__':
    for epoch in range(config.EPOCHS):
        avg_loss = train_fn(train_dataloader, model, device, optimizer, scheduler)
        avg_acc = eval_fn(test_dataloader, model, device)
        print(f'Epoch: {epoch} Avg train loss: {avg_loss} Avg valid acc: {avg_acc}')
    torch.save(model.state_dict(), 'model_weights01.pth')