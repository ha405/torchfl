import torch.nn as nn


class TwoNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 1,
                 hidden: int = 200, input_size: int = 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * input_size * input_size, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
