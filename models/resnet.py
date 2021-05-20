

"""
Reference:
Yerlan Idelbayev: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AdversarialDefensiveModel, generate_weights



__all__ = ["ResNet", "resnet20", "resnet32", "resnet56", "resnet110"]

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, 3, 
                    stride=stride, padding=padding, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)



class BasicBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels,
        stride=1, shortcut=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.shortway = shortcut

    def forward(self, x):
        temp = self.activation(self.bn1(self.conv1(x)))
        outs = self.bn2(self.conv2(temp))
        outs2 = x if self.shortway is None else self.shortway(x)
        return self.activation(outs + outs2)


class ResNet(AdversarialDefensiveModel):

    def __init__(
        self, layers, num_classes=10,
        scale=10., block=BasicBlock
    ):
        super(ResNet, self).__init__()


        self.conv0 = conv3x3(3, 16)
        self.bn0 = nn.BatchNorm2d(16)
        self.act0 = nn.PReLU()
        self.cur_channels = 16

        self.layer1 = self._make_layer(block, 16, layers[0], 1) # 16 x 32 x 32
        self.layer2 = self._make_layer(block, 32, layers[1], 2) # 32 x 16 x 16
        self.layer3 = self._make_layer(block, 64, layers[2], 2) # 64 x 8 x 8

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes, bias=False)

        # orthogonal classifier construction
        _weights = generate_weights(64)[:num_classes] * scale
        self.fc.weight.data.copy_(_weights)
        self.fc.requires_grad_(False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, block, out_channels, num_blocks, stride):

        shortcut = None
        if stride != 1 and out_channels != self.cur_channels:
            shortcut = conv1x1(self.cur_channels, out_channels, stride)
        
        layers = [block(self.cur_channels, out_channels, stride, shortcut)]
        self.cur_channels = out_channels
        for _ in range(num_blocks-1):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.act0(self.bn0(self.conv0(inputs)))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        features = self.avg_pool(l3).flatten(start_dim=1)
        logits = self.fc(features)
        if self.training:
            return features, logits
        elif self.attacking:
            return features
        else:
            return logits


def resnet20(num_classes=10, scale=10.):
    return ResNet([3, 3, 3], num_classes=num_classes, scale=scale)


def resnet32(num_classes=10, scale=10.):
    return ResNet([5, 5, 5], num_classes=num_classes, scale=scale)


def resnet56(num_classes=10, scale=10.):
    return ResNet([9, 9, 9], num_classes=num_classes, scale=scale)


def resnet110(num_classes=10, scale=10.):
    return ResNet([18, 18, 18], num_classes=num_classes, scale=scale)
        






















