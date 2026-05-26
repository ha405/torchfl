import torchvision
import torchvision.transforms as T

dataset_class  = torchvision.datasets.FashionMNIST
num_classes    = 10
input_channels = 1


def train_transforms():
    return T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])


def test_transforms():
    return T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
