import torchvision
import torchvision.transforms as T

dataset_class  = torchvision.datasets.MNIST
num_classes    = 10
input_channels = 1


def train_transforms():
    return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


def test_transforms():
    return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
