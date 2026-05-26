import torch.nn as nn

from .simple_cnn import SimpleCNN
from .lenet5 import LeNet5
from .mobilenetv2 import MobileNetV2
from .twonn import TwoNN
from .resnet import ResNet

_REGISTRY = {
    "simplecnn":   SimpleCNN,
    "lenet5":      LeNet5,
    "lenet":       LeNet5,
    "mobilenetv2": MobileNetV2,
    "mobilenet":   MobileNetV2,
    "twonn":       TwoNN,
    "2nn":         TwoNN,
    "resnet":      ResNet,
    "resnet10":    ResNet,
    "resnet18":    ResNet,
}


def get_model(config) -> nn.Module:
    from datasets import input_channels
    in_ch = input_channels(config.dataset_name)
    name  = getattr(config, "model_name", "SimpleCNN").lower()
    cls   = _REGISTRY.get(name)
    if cls is None:
        names = [c.__name__ for c in dict.fromkeys(_REGISTRY.values())]
        raise ValueError(f"Unknown model {config.model_name!r}. Available: {names}")
    if cls is TwoNN:
        input_size = 28 if in_ch == 1 else 32
        return cls(num_classes=config.num_classes, input_channels=in_ch,
                   hidden=getattr(config, "hidden_size", 200),
                   input_size=input_size).to(config.device)
    if cls is SimpleCNN:
        return cls(conv_channels=config.conv_channels,
                   num_classes=config.num_classes, input_channels=in_ch).to(config.device)
    return cls(num_classes=config.num_classes, input_channels=in_ch).to(config.device)
