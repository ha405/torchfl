import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from engine.pruning import get_current_sparsity, apply_weight_sparsity
from engine.data_cache import EvaluationCache
from engine.scheduler import get_lr


def _make_optimizer(params, config, lr: float):
    name = getattr(config, "optimizer", "adam").lower()
    wd   = getattr(config, "weight_decay", 0.0)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=getattr(config, "momentum", 0.9),
                         weight_decay=wd, nesterov=True)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    return optim.Adam(params, lr=lr, weight_decay=wd)


class FederatedClient:
    def __init__(self, client_id: int, train_dataloader, config, class_names=None):
        self.client_id   = client_id
        self.config      = config
        self.class_names = class_names
        self.device      = config.device

        # Pre-load all training batches to device to eliminate per-step PCIe transfers
        self.dataloader = [
            (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
            for x, y in train_dataloader
        ]
        self.dataset_size = sum(x.size(0) for x, _ in self.dataloader)

        from models import get_model
        self.model = get_model(self.config).to(self.device)

    def train(self, global_model, round_num: int = 0,
              scaffold_c: Optional[dict] = None,
              scaffold_ci: Optional[dict] = None) -> dict:
        algo = getattr(self.config, "fl_algorithm", "fedavg").lower()

        target = getattr(self.model, "_orig_mod", self.model)
        target.load_state_dict(global_model.state_dict())
        self.model.train()

        # fednova normalization is server-side; only fedprox/scaffold need the global snapshot
        if algo in ("fedprox", "scaffold"):
            global_state = {k: v.clone().float().to(self.device)
                            for k, v in global_model.state_dict().items()}

        lr        = get_lr(round_num, self.config)
        optimizer = _make_optimizer(self.model.parameters(), self.config, lr)
        criterion = nn.CrossEntropyLoss()

        total_steps  = len(self.dataloader) * self.config.local_epochs
        current_step = 0
        running_loss = torch.tensor(0.0, device=self.device)
        correct      = torch.tensor(0,   device=self.device)
        total        = 0

        # Pre-cast correction once so the per-step loop does zero dtype work
        scaffold_correction = None
        if algo == "scaffold" and scaffold_c is not None and scaffold_ci is not None:
            scaffold_correction = {
                k: (scaffold_c[k] - scaffold_ci[k]).float().to(self.device)
                for k in scaffold_c
            }

        mu   = getattr(self.config, "fedprox_mu", 0.01) if algo == "fedprox" else None
        clip = getattr(self.config, "grad_clip", None)

        for _ in range(self.config.local_epochs):
            for inputs, labels in self.dataloader:
                if inputs.size(0) <= 1:
                    continue

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = criterion(outputs, labels)

                if algo == "fedprox":
                    prox = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if name in global_state:
                            diff = param.float() - global_state[name]
                            prox = prox + diff.pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                loss.backward()

                if scaffold_correction is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in scaffold_correction:
                            param.grad.data.add_(scaffold_correction[name])

                if clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                optimizer.step()

                running_loss += loss.detach()
                _, predicted  = outputs.max(1)
                total        += labels.size(0)
                correct      += predicted.eq(labels).sum()

                if self.config.train_mode == "sparse":
                    apply_weight_sparsity(
                        self.model,
                        get_current_sparsity(current_step, total_steps,
                                             self.config.target_sparsity),
                    )
                current_step += 1

        if self.config.train_mode == "sparse":
            apply_weight_sparsity(self.model, self.config.target_sparsity)

        avg_loss  = running_loss.item() / current_step if current_step > 0 else 0.0
        train_acc = 100.0 * correct.item() / total if total > 0 else 0.0
        trained   = getattr(self.model, "_orig_mod", self.model)

        result = {
            "model":        trained,
            "metrics":      {"loss": avg_loss, "accuracy": train_acc},
            "dataset_size": self.dataset_size,
            "num_steps":    current_step,
        }

        if algo == "scaffold" and scaffold_c is not None and scaffold_ci is not None:
            K       = max(current_step, 1)
            delta_c = {}
            for name, param in trained.named_parameters():
                if name in scaffold_c:
                    w0     = global_state[name]
                    wi     = param.data.float()
                    ci     = scaffold_ci[name].float()
                    c      = scaffold_c[name].float()
                    ci_new = ci - c + (w0 - wi) / (K * lr)
                    delta_c[name] = (ci_new - ci).to(param.dtype)
            result["delta_c"] = delta_c

        return result

    def evaluate_on_test(self, model, cache: EvaluationCache) -> dict:
        target = getattr(self.model, "_orig_mod", self.model)
        if model is not self.model:
            target.load_state_dict(model.state_dict())

        self.model.eval()
        class_acc = {}
        with torch.no_grad():
            for c in range(self.config.num_classes):
                inputs, labels = cache.get_class_data(c)
                if inputs.shape[0] == 0:
                    continue
                correct, total = torch.tensor(0, device=self.device), 0
                for i in range(0, inputs.shape[0], self.config.batch_size):
                    out      = self.model(inputs[i:i + self.config.batch_size])
                    _, pred  = torch.max(out, 1)
                    lbl      = labels[i:i + self.config.batch_size]
                    total   += lbl.size(0)
                    correct += (pred == lbl).sum()
                class_acc[c] = round(100.0 * correct.item() / total, 4) if total else 0.0
        return class_acc
