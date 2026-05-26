from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List

from . import mnist, fashionmnist, cifar10, cifar100

_REGISTRY = {
    "MNIST":        mnist,
    "FASHIONMNIST": fashionmnist,
    "FMNIST":       fashionmnist,
    "CIFAR10":      cifar10,
    "CIFAR100":     cifar100,
}


def _get(name: str):
    key = name.upper().replace("-", "").replace("_", "")
    mod = _REGISTRY.get(key)
    if mod is None:
        available = [k for k in _REGISTRY if k != "FMNIST"]
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}")
    return mod


def num_classes(dataset_name: str) -> int:
    return _get(dataset_name).num_classes


def input_channels(dataset_name: str) -> int:
    return _get(dataset_name).input_channels


def get_dataset(config):
    mod      = _get(config.dataset_name)
    trainset = mod.dataset_class(root=config.data_root, train=True,
                                  download=True, transform=mod.train_transforms())
    testset  = mod.dataset_class(root=config.data_root, train=False,
                                  download=True, transform=mod.test_transforms())
    train_labels = get_labels(trainset)
    test_labels  = get_labels(testset)
    return (Subset(trainset, np.where(train_labels < config.num_classes)[0]),
            Subset(testset,  np.where(test_labels  < config.num_classes)[0]))


def get_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "targets"):
        return np.array(dataset.dataset.targets)[np.array(dataset.indices)]
    return np.array([y for _, y in dataset])


def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    indices = np.random.permutation(len(dataset))
    return [s.tolist() for s in np.array_split(indices, num_clients)]


def partition_dirichlet(dataset, num_clients: int, alpha: float, num_classes_: int) -> List[List[int]]:
    labels         = get_labels(dataset)
    client_indices = [[] for _ in range(num_clients)]
    # retry until every client has at least 10 samples
    while min(len(c) for c in client_indices) < 10:
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes_):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions /= proportions.sum()
            splits = np.split(idx_k, (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1])
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
    return client_indices


def get_dataloader(dataset, indices: List[int], config, shuffle: bool = True) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, pin_memory="cuda" in str(config.device))


def get_test_dataloader(testset, config) -> DataLoader:
    return DataLoader(testset, batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers, pin_memory="cuda" in str(config.device))
