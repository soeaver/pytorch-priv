from __future__ import division

""" 
Creates a SE-MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
And
Jie Hu, Li Shen, Gang Sun. (2017)
Squeeze-and-Excitation Networks
Copyright (c) Yang Lu, 2017
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['se_mobilenet', 'se_mobilenet_2', 'se_mobilenet_1', 'se_mobilenet_075', 'se_mobilenet_05', 
           'se_mobilenet_025']


class SEDepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(SEDepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes, planes // 16)
        self.fc2 = nn.Linear(planes // 16, planes)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        se = self.global_avg(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)

        out = out * se.expand_as(out)

        return out


class SE_MobileNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(SE_MobileNet, self).__init__()

        block = SEDepthWiseBlock

        self.conv1 = nn.Conv2d(3, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_mobilenet(widen_factor=1.0, num_classes=1000):
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=widen_factor, num_classes=num_classes)
    return model


def se_mobilenet_2():
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=2.0, num_classes=1000)
    return model


def se_mobilenet_1():
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=1.0, num_classes=1000)
    return model


def se_mobilenet_075():
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=0.75, num_classes=1000)
    return model


def se_mobilenet_05():
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=0.5, num_classes=1000)
    return model


def se_mobilenet_025():
    """
    Construct SE_MobileNet.
    """
    model = SE_MobileNet(widen_factor=0.25, num_classes=1000)
    return model
