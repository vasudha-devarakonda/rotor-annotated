from collections import namedtuple, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from .utils import *


import torch
import torch.nn as nn

# Reusing your BasicConv2d
class BasicConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, bias=False, **kwargs))
        self.add_module('bn_relu', BatchNorm2dAndReLU(out_channels, eps=1e-05))


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Sequential):
    def __init__(self, num_classes=100):
        super().__init__()


        self.add_module("prelayer", nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=False),
        ))


        self.add_module("maxpool1", nn.MaxPool2d(3, stride=2, padding=1))


        self.add_module('a3', Inception(192, 64, 96, 128, 16, 32, 32))
        self.add_module('b3', Inception(256, 128, 128, 192, 32, 96, 64))

        self.add_module("maxpool3", nn.MaxPool2d(3, stride=2, padding=1))

     
        self.add_module('a4', Inception(480, 192, 96, 208, 16, 48, 64))
        self.add_module('b4', Inception(512, 160, 112, 224, 24, 64, 64))
        self.add_module('c4', Inception(512, 128, 128, 256, 24, 64, 64))
        self.add_module('d4', Inception(512, 112, 144, 288, 32, 64, 64))
        self.add_module('e4', Inception(528, 256, 160, 320, 32, 128, 128))

        self.add_module("maxpool4", nn.MaxPool2d(3, stride=2, padding=1))

 
        self.add_module('a5', Inception(832, 256, 160, 320, 32, 128, 128))
        self.add_module('b5', Inception(832, 384, 192, 384, 48, 128, 128))


        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('dropout', nn.Dropout2d(0.4))
        self.add_module('flatten', nn.Flatten())
        self.add_module('fc', nn.Linear(1024, num_classes))


def googlenet(num_classes=100):
    return GoogleNet(num_classes=num_classes)
