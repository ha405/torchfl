import torch
import torch.nn as nn


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1      = _conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = _conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1   = self._make_layer(64,  stride=1)
        self.block2   = self._make_layer(128, stride=2)
        self.block3   = self._make_layer(192, stride=2)
        self.block4   = self._make_layer(256, stride=2)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(256, num_classes)

    def _make_layer(self, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layer = nn.Sequential(_BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        return layer

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))
