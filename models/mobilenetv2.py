import torch.nn as nn


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        from torchvision.models import mobilenet_v2
        base  = mobilenet_v2(weights=None)
        first = base.features[0][0]
        # stride 1 for small inputs (32x32); original stride 2 over-downsamples
        base.features[0][0] = nn.Conv2d(input_channels, first.out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False)
        self.features   = base.features
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(base.last_channel, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))
