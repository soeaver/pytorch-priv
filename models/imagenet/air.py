from __future__ import division

""" 
Aligned Inception ResNet
Copyright (c) Yang Lu, 2017
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['air14_1x64d', 'air26_1x64d', 'air50_1x64d', 'air101_1x64d', 'air152_1x64d']


class AIRBottleneck(nn.Module):
    """
    AIRBottleneck bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super(AIRBottleneck, self).__init__()	

        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv1_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes // 2)
        self.conv2_2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(planes // 2)
        self.conv2_3 = nn.Conv2d(planes // 2,  planes // 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_concat = nn.BatchNorm2d(planes + (planes // 2))

        self.conv = nn.Conv2d(planes + (planes // 2), planes*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.bn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.bn_concat(out)
        out = self.relu(out)
  
        out = self.conv(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AIR(nn.Module):
    def __init__(self, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., [3, 4, 23, 3]
            num_classes: number of classes
        """
        super(AIR, self).__init__()
        block = AIRBottleneck

        self.inplanes = baseWidth

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth)
        else:
            self.conv1 = nn.Conv2d(3, baseWidth // 2, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth // 2)
            self.conv2 = nn.Conv2d(baseWidth // 2, baseWidth // 2, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(baseWidth // 2)
            self.conv3 = nn.Conv2d(baseWidth // 2, baseWidth, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(baseWidth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, baseWidth, layers[0])
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, baseWidth * 4, layers[2], 2)
        self.layer4 = self._make_layer(block, baseWidth * 8, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct AIR
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def air(baseWidth=64, head7x7=False, layers=(3, 4, 23, 3), num_classes=1000):
    """
    Construct AIR.
    (2, 2, 2, 2) for air26
    (3, 4, 6, 3) for air50
    (3, 4, 23, 3) for air101
    (3, 8, 36, 3) for air152
    note: The actual depth of air is not the same as its name.
    """
    model = AIR(baseWidth=baseWidth, head7x7=head7x7, layers=layers, num_classes=num_classes)
    return model


def air14_1x64d():
    model = AIR(baseWidth=64, head7x7=False, layers=(1, 1, 1, 1), num_classes=1000)
    return model


def air26_1x64d():
    model = AIR(baseWidth=64, head7x7=False, layers=(2, 2, 2, 2), num_classes=1000)
    return model


def air50_1x64d():
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def air101_1x64d():
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 4, 23, 3), num_classes=1000)
    return model


def air152_1x64d():
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 8, 36, 3), num_classes=1000)
    return model
