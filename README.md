# torchfl

A clean, extensible federated learning framework built for PyTorch research.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`

---

## Quick Start

**Single run via CLI:**

```bash
python main.py --dataset CIFAR10 --model ResNet --algorithm fedavg \
               --num_clients 10 --num_rounds 100 --local_epochs 5 \
               --partition dirichlet --alpha 0.1 --scheduler cosine \
               --output_dir ./results/my_run
```

**Single run via config file:**

```bash
# edit config.yaml, then:
python main.py --config_file config.yaml
```

CLI flags always override config file values.

**Batch experiments:**

```bash
python run_suite.py                    # run all experiments in EXPERIMENTS list
python run_suite.py --filter scaffold  # only experiments whose name contains "scaffold"
python run_suite.py --dry-run          # print commands without running
python run_suite.py --test             # 1-round smoke test for each experiment
python run_suite.py --device cpu       # override device for all runs
```

**Regenerate figures for a completed run:**

```bash
python plot.py results/my_run
python plot.py results/run_a results/run_b          # multiple runs at once
python plot.py results/my_run --type training_curves
```

---

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `fedavg` | Weighted FedAvg - aggregation weighted by client dataset size |
| `fedprox` | FedAvg + proximal term `(mu/2)||w - w_global||^2` added to client loss |
| `fednova` | Normalizes each client's update by its local step count before aggregation; corrects for heterogeneous training intensity |
| `scaffold` | Variance-reduced FedAvg with per-client control variates that correct client drift |

---

## Output Structure

```
results/my_run/
  config.yaml               exact config used
  metrics.json              per-round accuracy, loss, client metrics
  metrics.csv               same data in tabular form
  run_info.json             system info, duration, final accuracy, git commit
  checkpoints/
    initialization.pt       global model before training
    best_model.pt           highest accuracy checkpoint
    checkpoint_round_N.pt   periodic checkpoints (checkpoint_every)
    round_N/
      client_K_model.pt     per-client weights after round N
  logs/
    training_log.txt        full per-round and per-class breakdown
  figures/
    training_curves.png
    client_accuracy.png
    data_distribution.png
    class_accuracy_heatmap.png
  partitions/
    client_partitions.json  index lists per client
    labels.npy              full label array for visualization
```

---

## GPU Memory

torchfl pre-loads all training data and the full test set to the GPU at startup, and keeps one model copy per selected client in memory during each round. This is what makes it fast, but it means GPU memory scales with clients, model size, and dataset size.

**If you hit OOM:**

| Cause | Fix |
|-------|-----|
| Too many concurrent clients | Lower `fraction_fit` (e.g. `0.2`) or `num_clients` |
| Large dataset pre-loaded to GPU | Lower `batch_size` — fewer batches means less memory per client |
| Large model | Switch to a smaller model (e.g. `SimpleCNN` instead of `ResNet`) |
| Many clients each holding a model | Lower `num_clients` |
| CIFAR100 with many clients | Use `fraction_fit: 0.1` — only 10% train per round |

**General guidance by GPU size:**

| VRAM | Recommended setup |
|------|-------------------|
| 4 GB | MNIST/FashionMNIST, TwoNN or LeNet5, `num_clients <= 10` |
| 8 GB | CIFAR10, SimpleCNN or ResNet, `num_clients <= 20` |
| 16 GB+ | CIFAR100, any model, large client counts |

If GPU memory is a hard constraint, set `device: cpu` — training is slower but memory is not a bottleneck.

