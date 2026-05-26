"""
FastFL multi-experiment runner.

Define your experiments in the EXPERIMENTS list below, then run:

  python run_suite.py                      # run everything
  python run_suite.py --device cpu         # override device for all runs
  python run_suite.py --filter scaffold    # only runs whose name contains "scaffold"
  python run_suite.py --dry-run            # print commands without executing
  python run_suite.py --test               # 1-round smoke test

Each experiment is a dict with:
  "name"  - unique identifier (used in dry-run output and filtering)
  "cli"   - dict of CLI arguments passed to main.py

Any argument accepted by main.py can appear in "cli".
See `python main.py --help` for the full list.
"""

import argparse
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {
        "name": "cifar10_cnn_iid_fedavg",
        "cli": {
            "dataset": "CIFAR10", "model": "SimpleCNN",
            "partition": "iid", "algorithm": "fedavg",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_cnn_iid_fedavg",
        },
    },
    {
        "name": "cifar10_cnn_noniid_fedavg",
        "cli": {
            "dataset": "CIFAR10", "model": "SimpleCNN",
            "partition": "dirichlet", "alpha": 0.1, "algorithm": "fedavg",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_cnn_noniid_fedavg",
        },
    },
    {
        "name": "cifar10_cnn_noniid_fedprox",
        "cli": {
            "dataset": "CIFAR10", "model": "SimpleCNN",
            "partition": "dirichlet", "alpha": 0.1, "algorithm": "fedprox",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_cnn_noniid_fedprox",
        },
    },
    {
        "name": "cifar10_cnn_noniid_scaffold",
        "cli": {
            "dataset": "CIFAR10", "model": "SimpleCNN",
            "partition": "dirichlet", "alpha": 0.1, "algorithm": "scaffold",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_cnn_noniid_scaffold",
        },
    },
    {
        "name": "cifar10_resnet_iid_fedavg",
        "cli": {
            "dataset": "CIFAR10", "model": "ResNet",
            "partition": "iid", "algorithm": "fedavg",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_resnet_iid_fedavg",
        },
    },
    {
        "name": "fmnist_lenet_iid_fedavg",
        "cli": {
            "dataset": "FashionMNIST", "model": "LeNet5",
            "partition": "iid", "algorithm": "fedavg",
            "num_rounds": 100, "num_clients": 10,
            "lr": 0.001, "scheduler": "warmup_cosine",
            "output_dir": "./results/fmnist_lenet_iid_fedavg",
        },
    },
    {
        "name": "cifar10_cnn_partial_fednova",
        "cli": {
            "dataset": "CIFAR10", "model": "SimpleCNN",
            "partition": "dirichlet", "alpha": 0.1, "algorithm": "fednova",
            "num_rounds": 100, "num_clients": 20, "fraction_fit": 0.5,
            "lr": 0.001, "scheduler": "cosine",
            "output_dir": "./results/cifar10_cnn_partial_fednova",
        },
    },
]


def _build_cmd(exp: dict, device: str, test: bool) -> list:
    cli = dict(exp["cli"])
    if test:
        cli["num_rounds"] = 1
        if "output_dir" in cli:
            cli["output_dir"] = cli["output_dir"] + "_test"
    cli["device"] = device

    cmd = [sys.executable, "main.py"]
    for k, v in cli.items():
        if v is None:
            continue
        cmd += ["--" + k, str(v)]
    return cmd


def _run(cmd: list, name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}")
    t0      = time.time()
    result  = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    ok      = result.returncode == 0
    print(f"\n  [{'OK' if ok else 'FAILED'}]  {name}  -  {elapsed:.0f}s")
    return ok


def main():
    parser = argparse.ArgumentParser(description="FastFL experiment suite")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--filter",  default=None,
                        help="Only run experiments whose name contains this string")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--test",    action="store_true",
                        help="1-round smoke test for each experiment")
    args = parser.parse_args()

    to_run = [e for e in EXPERIMENTS
              if args.filter is None or args.filter in e["name"]]

    if not to_run:
        print(f"No experiments match filter {args.filter!r}.")
        sys.exit(0)

    print(f"\nFastFL Suite  |  {len(to_run)} experiment(s)"
          f"{'  [DRY RUN]' if args.dry_run else ''}"
          f"{'  [TEST MODE]' if args.test else ''}")

    failures = []
    t_start  = time.time()

    for exp in to_run:
        cmd = _build_cmd(exp, args.device, args.test)
        if args.dry_run:
            print(f"\n  {exp['name']}")
            print(f"  {' '.join(cmd)}")
            continue
        if not _run(cmd, exp["name"]):
            failures.append(exp["name"])

    elapsed = time.time() - t_start
    if not args.dry_run:
        print(f"\n{'='*70}")
        print(f"  Done  -  {elapsed/60:.1f} min")
        if failures:
            print("  Failed:")
            for f in failures:
                print(f"    x  {f}")
        else:
            print(f"  All {len(to_run)} experiments completed successfully.")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
