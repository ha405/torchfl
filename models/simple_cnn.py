import torch.nn as nn
from typing import List


class SimpleCNN(nn.Module):
    def __init__(self, conv_channels: List[int] = None, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_channels[0], conv_channels[1], 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_channels[1], conv_channels[2], 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        # 28x28 inputs pool to 3x3; 32x32 inputs pool to 4x4
        spatial = 3 if input_channels == 1 else 4
        self.classifier = nn.Linear(conv_channels[2] * spatial * spatial, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
