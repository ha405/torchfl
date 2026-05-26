import argparse
import sys
import os
import json
import time
import torch
import numpy as np
import random
import shutil

from tqdm import tqdm

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())

from engine.config import ExperimentConfig
from datasets import (
    get_dataset, get_test_dataloader, get_dataloader,
    partition_iid, partition_dirichlet, get_labels,
)
from models import get_model
from engine.scheduler import get_lr
from engine.utils import load_latest_checkpoint, save_checkpoint, evaluate_model
from engine.data_cache import EvaluationCache
from engine.metrics import MetricsTracker, RunInfo
from engine.visualization import generate_all_plots
from federated.client import FederatedClient
from federated.server import FederatedServer


W = 72

def _box(lines, width=W):
    inner = width - 4
    top   = "+" + "-" * (width - 2) + "+"
    bot   = "+" + "-" * (width - 2) + "+"
    rows  = [f"|  {l:<{inner}}  |" for l in lines]
    return "\n".join([top] + rows + [bot])

def _sep(char="-", width=W): return char * width

def _kv(label, value, lw=26): return f"  {label:<{lw}} {value}"


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.counter   = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True

    def setup(self):
        cfg = self.config
        self._set_seed()

        if not cfg.resume and os.path.exists(cfg.output_dir):
            shutil.rmtree(cfg.output_dir, ignore_errors=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.dirs = {
            "checkpoints": os.path.join(cfg.output_dir, "checkpoints"),
            "logs":        os.path.join(cfg.output_dir, "logs"),
            "figures":     os.path.join(cfg.output_dir, "figures"),
            "partitions":  os.path.join(cfg.output_dir, "partitions"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        import yaml
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg.__dict__, f, default_flow_style=False, sort_keys=False)

        tqdm.write(f"\n  Loading {cfg.dataset_name}...")
        trainset, testset = get_dataset(cfg)
        self.testloader   = get_test_dataloader(testset, cfg)

        if hasattr(trainset, "classes"):
            self.class_names = list(trainset.classes)
        elif hasattr(trainset, "dataset") and hasattr(trainset.dataset, "classes"):
            self.class_names = list(trainset.dataset.classes)
        else:
            self.class_names = [str(i) for i in range(cfg.num_classes)]

        self.evaluation_cache = EvaluationCache(self.testloader, cfg.device, cfg.num_classes)

        if cfg.partition_method == "iid":
            client_indices = partition_iid(trainset, cfg.num_clients)
        elif cfg.partition_method == "dirichlet":
            client_indices = partition_dirichlet(
                trainset, cfg.num_clients, cfg.dirichlet_alpha, cfg.num_classes
            )
        else:
            raise ValueError(f"Unknown partition_method: {cfg.partition_method!r}")

        with open(os.path.join(self.dirs["partitions"], "client_partitions.json"), "w") as f:
            json.dump(client_indices, f)
        all_labels = get_labels(trainset)
        np.save(os.path.join(self.dirs["partitions"], "labels.npy"), all_labels)

        sizes = [len(idx) for idx in client_indices if idx]
        tqdm.write(
            f"  Partitioned {len(trainset):,} samples -> {len(sizes)} clients  "
            f"(min {min(sizes):,} / max {max(sizes):,})"
        )

        self.clients = [
            FederatedClient(i, get_dataloader(trainset, idx, cfg), cfg, self.class_names)
            for i, idx in enumerate(client_indices) if idx
        ]

        self.global_model = get_model(cfg)
        params = sum(p.numel() for p in self.global_model.parameters())
        tqdm.write(f"  Model: {cfg.model_name}  ({params:,} parameters)")

        torch.save(
            self.global_model.state_dict(),
            os.path.join(self.dirs["checkpoints"], "initialization.pt"),
        )

        self.server  = FederatedServer(
            self.global_model, cfg, self.class_names, self.evaluation_cache
        )
        self.tracker = MetricsTracker(cfg.output_dir)
        if cfg.resume:
            self.tracker.load()
        self.run_info = RunInfo(cfg.output_dir, cfg)

    def run(self):
        cfg        = self.config
        algo_str   = cfg.fl_algorithm.upper()
        sched_str  = cfg.lr_scheduler if cfg.lr_scheduler != "constant" else "fixed LR"
        sparse_str = (f"sparse ({cfg.target_sparsity*100:.0f}%)"
                      if cfg.train_mode == "sparse" else "dense")
        part_str   = (f"Dirichlet a={cfg.dirichlet_alpha}"
                      if cfg.partition_method == "dirichlet" else "IID")
        frac_str   = (f"{int(cfg.fraction_fit*100)}% of clients/round"
                      if cfg.fraction_fit < 1.0 else "all clients/round")

        tqdm.write("\n" + _box([
            f"FastFL  |  {cfg.dataset_name}  |  {cfg.model_name}",
            f"Algorithm: {algo_str}   Partition: {part_str}",
            f"Clients: {cfg.num_clients}  ({frac_str})",
            f"Rounds: {cfg.num_rounds}   Local epochs: {cfg.local_epochs}",
            f"Optim: {cfg.optimizer.upper()}   Scheduler: {sched_str}   Training: {sparse_str}",
            f"Output: {cfg.output_dir}",
        ]))

        start_round = 0
        if cfg.resume:
            start_round = load_latest_checkpoint(self.global_model, cfg)
        if start_round >= cfg.num_rounds:
            tqdm.write("  Training already complete.")
            return

        log_path  = os.path.join(self.dirs["logs"], "training_log.txt")
        final_acc = 0.0
        best_acc  = 0.0
        best_path = os.path.join(self.dirs["checkpoints"], "best_model.pt")

        early_stop = None
        if cfg.early_stopping:
            early_stop = EarlyStopping(cfg.early_stopping_patience,
                                       cfg.early_stopping_min_delta)

        with open(log_path, "a" if cfg.resume else "w") as log_f:
            pbar = tqdm(
                range(start_round, cfg.num_rounds),
                desc=f"  {algo_str}",
                unit="round",
                dynamic_ncols=True,
                bar_format="{desc} {bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}]",
            )

            for round_num in pbar:
                t0 = time.time()
                log_f.write(f"\n{'='*70}\nROUND {round_num+1}/{cfg.num_rounds}"
                            f"  LR={get_lr(round_num, cfg):.6f}\n{'='*70}\n")

                client_metrics, client_test_metrics, n_selected = \
                    self.server.orchestrate_round(round_num, self.clients, log_f)

                acc, loss, class_acc = evaluate_model(
                    self.global_model, self.evaluation_cache, cfg,
                    class_names=self.class_names,
                    title=f"Round {round_num+1}",
                    log_file=log_f,
                )
                final_acc = acc
                elapsed   = time.time() - t0

                if acc > best_acc:
                    best_acc = acc
                    if cfg.save_best_model:
                        torch.save({
                            "round": round_num + 1,
                            "accuracy": best_acc,
                            "model_state_dict": self.global_model.state_dict(),
                        }, best_path)

                pbar.set_postfix(
                    acc=f"{acc:.2f}%", loss=f"{loss:.4f}",
                    best=f"{best_acc:.2f}%", lr=f"{get_lr(round_num, cfg):.5f}",
                    selected=n_selected,
                    refresh=True,
                )

                client_summary = "  ".join(
                    f"c{cid}:{m['accuracy']:.1f}%"
                    for cid, m in sorted(client_metrics.items())
                )
                tqdm.write(
                    f"  Round {round_num+1:>4}/{cfg.num_rounds}"
                    f"  |  acc {acc:6.2f}%  loss {loss:.4f}"
                    f"  |  {elapsed:.1f}s  ({n_selected}/{cfg.num_clients} clients)"
                )
                tqdm.write(f"  Train:  {client_summary}")

                log_f.write("\n[Client test accuracy]\n")
                for cid in sorted(client_test_metrics):
                    parts = "  ".join(
                        f"{self.class_names[c] if c < len(self.class_names) else c}:"
                        f"{v:.1f}%" for c, v in sorted(client_test_metrics[cid].items())
                    )
                    log_f.write(f"  client_{cid} | {parts}\n")
                log_f.flush()

                self.tracker.log_round(
                    round_num, acc, loss,
                    global_class_acc=class_acc,
                    client_train_metrics=client_metrics,
                    client_test_metrics=client_test_metrics,
                    round_time=elapsed,
                )
                self.tracker.save()

                if cfg.checkpoint_every > 0 and (round_num + 1) % cfg.checkpoint_every == 0:
                    save_checkpoint(
                        self.global_model, round_num + 1, cfg, self.dirs["checkpoints"]
                    )

                if early_stop and early_stop.step(acc):
                    tqdm.write(
                        f"\n  [Early stopping] No improvement for "
                        f"{cfg.early_stopping_patience} rounds. Stopping at round {round_num+1}."
                    )
                    break

        self.tracker.save_csv()
        self.run_info.finish(final_accuracy=final_acc)

        dur = self.run_info.data.get("duration_seconds", 0)
        tqdm.write("\n" + _sep())
        tqdm.write(_kv("Final accuracy:",  f"{final_acc:.2f}%"))
        tqdm.write(_kv("Best accuracy:",   f"{best_acc:.2f}%"))
        tqdm.write(_kv("Total time:",      f"{dur/60:.1f} min"))
        if cfg.save_best_model and os.path.exists(best_path):
            tqdm.write(_kv("Best model saved:", best_path))
        tqdm.write(_sep())

        tqdm.write("\n  Generating figures...")
        generate_all_plots(cfg.output_dir, class_names=self.class_names)
        tqdm.write("")


def parse_args():
    p = argparse.ArgumentParser(
        description="FastFL - Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config_file",    type=str,   help="YAML (or JSON) config file")
    p.add_argument("--device",         type=str)
    p.add_argument("--seed",           type=int)
    p.add_argument("--output_dir",     type=str)
    p.add_argument("--dataset",        type=str,   help="MNIST / FashionMNIST / CIFAR10 / CIFAR100")
    p.add_argument("--model",          type=str,   help="SimpleCNN / LeNet5 / ResNet / MobileNetV2 / TwoNN")
    p.add_argument("--num_rounds",     type=int)
    p.add_argument("--num_clients",    type=int)
    p.add_argument("--local_epochs",   type=int)
    p.add_argument("--batch_size",     type=int)
    p.add_argument("--lr",             type=float)
    p.add_argument("--partition",      type=str,   help="iid / dirichlet")
    p.add_argument("--alpha",          type=float, help="Dirichlet alpha")
    p.add_argument("--algorithm",      type=str,   help="fedavg / fedprox / fednova / scaffold")
    p.add_argument("--optimizer",      type=str,   help="adam / adamw / sgd")
    p.add_argument("--scheduler",      type=str,   help="constant / cosine / step / warmup_cosine / exponential / polynomial")
    p.add_argument("--fraction_fit",   type=float, help="Fraction of clients per round")
    p.add_argument("--train_mode",     type=str,   choices=["dense", "sparse"])
    p.add_argument("--sparsity",       type=float)
    p.add_argument("--grad_clip",      type=float)
    p.add_argument("--resume",         action="store_true")
    return p.parse_args()


def update_config(config: ExperimentConfig, args) -> ExperimentConfig:
    if args.config_file:
        with open(args.config_file) as f:
            if args.config_file.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)

    mapping = {
        "output_dir":  "output_dir",   "device":      "device",
        "seed":        "seed",         "dataset":     "dataset_name",
        "partition":   "partition_method",
        "alpha":       "dirichlet_alpha",
        "model":       "model_name",   "num_rounds":  "num_rounds",
        "num_clients": "num_clients",  "local_epochs": "local_epochs",
        "batch_size":  "batch_size",   "lr":          "learning_rate",
        "algorithm":   "fl_algorithm", "optimizer":   "optimizer",
        "scheduler":   "lr_scheduler", "fraction_fit": "fraction_fit",
        "train_mode":  "train_mode",   "sparsity":    "target_sparsity",
        "grad_clip":   "grad_clip",
    }
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, cfg_name, val)

    if args.resume:
        config.resume = True
    return config


def main():
    args   = parse_args()
    config = update_config(ExperimentConfig(), args)
    runner = ExperimentRunner(config)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
