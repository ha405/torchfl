import torch
from torch.utils.data import DataLoader


class EvaluationCache:
    def __init__(self, testloader: DataLoader, device: str, num_classes: int):
        all_inputs, all_labels = [], []
        for inputs, labels in testloader:
            all_inputs.append(inputs)
            all_labels.append(labels)

        self.inputs      = torch.cat(all_inputs).to(device)
        self.labels      = torch.cat(all_labels).to(device)
        self.device      = device
        self.num_classes = num_classes

        self._class_inputs = {}
        self._class_labels = {}
        for c in range(num_classes):
            idx = (self.labels == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                self._class_inputs[c] = self.inputs[idx]
                self._class_labels[c] = self.labels[idx]
            else:
                self._class_inputs[c] = torch.empty(0, *self.inputs.shape[1:], device=device)
                self._class_labels[c] = torch.empty(0, dtype=torch.long, device=device)

    def get_class_data(self, target_class: int):
        return self._class_inputs[target_class], self._class_labels[target_class]

    def iterate_batches(self, batch_size: int = 256):
        n = self.inputs.shape[0]
        for i in range(0, n, batch_size):
            yield self.inputs[i:i + batch_size], self.labels[i:i + batch_size]

    def __len__(self):
        return self.inputs.shape[0]
