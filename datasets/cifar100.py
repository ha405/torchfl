import torchvision
import torchvision.transforms as T

dataset_class  = torchvision.datasets.CIFAR100
num_classes    = 100
input_channels = 3


def train_transforms():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])


def test_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
