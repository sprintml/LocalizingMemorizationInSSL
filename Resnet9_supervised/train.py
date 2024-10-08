import torch
import config
from model import ResNet9
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from engine import train_fn, eval_fn

Aug = 1
model = ResNet9(3, 10)

if Aug:
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    s = 1
    color_jitter = tt.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
    flip = tt.RandomHorizontalFlip()
    data_transforms = tt.Compose(
        [
            tt.RandomResizedCrop(size=32),
            tt.RandomApply([flip], p=0.5),
            tt.RandomApply([color_jitter], p=0.4),
            tt.RandomGrayscale(p=0.1), ToTensor(), Normalize(0.5, 0.5)
        ])
else:
    data_transforms = Compose([ToTensor(), Normalize(0.5, 0.5)])
datapath = './data'
traindataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                            transform=data_transforms)

testdataset = torchvision.datasets.CIFAR10(datapath, train=False, download=False,
                                           transform=data_transforms)
train_dataloader = torch.utils.data.DataLoader(traindataset, config.TRAIN_BATCH_SIZE, shuffle=True,
                                               num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(testdataset, config.VALID_BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True)
# if torch.cuda.is_available():
#    device = torch.device('cuda')
# else:
device = torch.device('cuda')

model = model.to(device)
steps = len(train_dataloader) * config.EPOCHS
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.LEARNING_RATE, epochs=config.EPOCHS,
                                                total_steps=steps)
if __name__ == '__main__':
    for epoch in range(config.EPOCHS):
        avg_loss = train_fn(train_dataloader, model, device, optimizer, scheduler)
        avg_acc = eval_fn(test_dataloader, model, device)
        print(f'Epoch: {epoch} Avg train loss: {avg_loss} Avg valid acc: {avg_acc}')
    torch.save(model.state_dict(), 'model_weights0.pth')
