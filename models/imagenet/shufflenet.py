from __future__ import division

""" 
Creates a ShuffleNet Model as defined in:
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun. (2017). 
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. 
Copyright (c) Yang Lu, 2017
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

__all__ = ['shufflenet', 'shufflenet_w2g3', 'shufflenet_w15g3', 'shufflenet_w1g3', 'shufflenet_w05g8']


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleAddBlock(nn.Module):
    def __init__(self, inplanes, outplanes, groups=3):
        super(ShuffleAddBlock, self).__init__()
        midplanes = outplanes // 4
        self.groups = groups

        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=1, padding=1, groups=midplanes, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = channel_shuffle(out, self.groups)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out


class ShuffleConcatBlock(nn.Module):
    def __init__(self, inplanes, outplanes, groups=3, first_group=True):
        super(ShuffleConcatBlock, self).__init__()
        midplanes = (outplanes - inplanes) // 4
        self.groups = groups
        self.first_group = first_group

        if self.first_group:
            self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=1, groups=groups, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=2, padding=1, groups=midplanes, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes - inplanes, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes - inplanes)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        downsample = self.avgpool(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.first_group:
            out = channel_shuffle(out, self.groups)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = torch.cat((downsample, out), 1)
        out = self.relu(out)

        return out


class ShuffleNet(nn.Module):
    def __init__(self, groups=3, widen_factor=1.0, num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(ShuffleNet, self).__init__()

        layers = (3, 7, 3)

        if groups == 1:
            stage_out_channels = [144, 288, 576]
        elif groups == 2:
            stage_out_channels = [200, 400, 800]
        elif groups == 3:
            stage_out_channels = [240, 480, 960]
        elif groups == 4:
            stage_out_channels = [272, 544, 1088]
        elif groups == 8:
            stage_out_channels = [384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        stage_out_channels = np.asarray(np.array(stage_out_channels) * widen_factor, dtype=np.int).tolist()
        self.groups = groups
        self.addblock = ShuffleAddBlock
        self.concatblock = ShuffleConcatBlock
        self.inplanes = int(24 * widen_factor)  # default 24

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_layer(stage_out_channels[0], layers[0], first_group=False)
        self.stage3 = self._make_layer(stage_out_channels[1], layers[1])
        self.stage4 = self._make_layer(stage_out_channels[2], layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, outplanes, blocks, first_group=True):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            outplanes: number of output channels
            blocks: number of blocks to be built
        Returns: a Module consisting of n sequential bottlenecks.
        """
        layers = []
        layers.append(self.concatblock(self.inplanes, outplanes, groups=self.groups, first_group=first_group))
        self.inplanes = outplanes
        for i in range(blocks):
            layers.append(self.addblock(self.inplanes, outplanes, groups=self.groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def shufflenet(groups=3, widen_factor=2.0, num_classes=1000):
    """
    Construct ShuffleNet.
    """
    model = ShuffleNet(groups=groups, widen_factor=widen_factor, num_classes=num_classes)
    return model


def shufflenet_w2g3():
    model = ShuffleNet(groups=3, widen_factor=2.0, num_classes=1000)
    return model


def shufflenet_w15g3():
    model = ShuffleNet(groups=3, widen_factor=1.5, num_classes=1000)
    return model


def shufflenet_w1g3():
    model = ShuffleNet(groups=3, widen_factor=1.0, num_classes=1000)
    return model


def shufflenet_w05g8():
    model = ShuffleNet(groups=8, widen_factor=0.5, num_classes=1000)
    return model
