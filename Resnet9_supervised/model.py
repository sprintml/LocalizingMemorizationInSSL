import torch.nn as nn
from base import Base
from collections import OrderedDict


def conv_bn_relu_pool(in_channels, out_channels, pool=False):
    net = nn.Sequential(
        OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)),
            ('batch', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    if pool:
        net = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)),
                ('batch', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
            ]))
    return net


class ResNet9(Base):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.prep = conv_bn_relu_pool(in_channels, 32, pool=True)
        self.layer1_head = conv_bn_relu_pool(32, 64, pool=True)
        self.layer1_residual1 = conv_bn_relu_pool(64, 64)
        self.layer1_residual2 = conv_bn_relu_pool(64, 64)
        self.layer2 = conv_bn_relu_pool(64, 128, pool=True)
        self.layer3_head = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_residual1 = conv_bn_relu_pool(256, 256)
        self.layer3_residual2 = conv_bn_relu_pool(256, 256)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.prep(x)

        x = self.layer1_head(x)

        a = x
        x = self.layer1_residual1(x)

        x = self.layer1_residual2(x) + a

        x = self.layer2(x)

        x = self.layer3_head(x)

        b = x
        x = self.layer3_residual1(x)

        x = self.layer3_residual2(x) + b

        x = self.classifier(x)

        return x
