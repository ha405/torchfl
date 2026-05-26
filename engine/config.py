import torch
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ExperimentConfig:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    output_dir: str = "./results/experiment"
    num_workers: int = 0

    dataset_name: str = "CIFAR10"
    data_root: str = "./data"
    num_classes: int = 10
    batch_size: int = 256

    model_name: str = "SimpleCNN"
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    hidden_size: int = 200

    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    fraction_fit: float = 1.0
    min_fit_clients: int = 1

    partition_method: str = "iid"
    dirichlet_alpha: float = 0.5

    fl_algorithm: str = "fedavg"
    fedprox_mu: float = 0.01

    optimizer: str = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None

    lr_scheduler: str = "constant"
    lr_min: float = 0.0
    lr_warmup_rounds: int = 0
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    lr_power: float = 1.0

    train_mode: str = "dense"
    target_sparsity: float = 0.99

    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    save_best_model: bool = True
    checkpoint_every: int = 1

    resume: bool = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save(self, path: str):
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str):
        import dataclasses
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f)
            else:
                import json
                data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)
