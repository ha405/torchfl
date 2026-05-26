import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        spatial = 5 if input_channels == 1 else 6
        self.classifier = nn.Sequential(
            nn.Linear(16 * spatial * spatial, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
