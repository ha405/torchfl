import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _figures_dir(output_dir):
    d = os.path.join(output_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d

def _load_rounds(output_dir):
    with open(os.path.join(output_dir, "metrics.json")) as f:
        return json.load(f)["rounds"]

def _save(fig, output_dir, name):
    out = os.path.join(_figures_dir(output_dir), name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(output_dir, title=None):
    rounds = _load_rounds(output_dir)
    xs     = [r["round"] for r in rounds]
    accs   = [r["global_accuracy"] for r in rounds]
    losses = [r["global_loss"] for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title or os.path.basename(output_dir), fontsize=12, fontweight="bold", y=1.01)

    ax1.plot(xs, accs, color=_PALETTE[0], linewidth=2)
    ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Global Test Accuracy")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if accs:
        peak = max(accs)
        ax1.axhline(peak, color=_PALETTE[0], linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.annotate(f"peak {peak:.2f}%", xy=(xs[accs.index(peak)], peak),
                     xytext=(8, -14), textcoords="offset points", fontsize=8, color=_PALETTE[0])

    ax2.plot(xs, losses, color=_PALETTE[1], linewidth=2)
    ax2.set_xlabel("Round"); ax2.set_ylabel("Loss"); ax2.set_title("Global Loss")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    fig.tight_layout()
    return _save(fig, output_dir, "training_curves.png")


def plot_client_accuracy(output_dir, title=None):
    rounds     = _load_rounds(output_dir)
    client_ids = sorted({cid for r in rounds for cid in r.get("clients", {})}, key=lambda x: int(x))
    if not client_ids:
        return None

    xs          = [r["round"] for r in rounds]
    client_accs = {cid: [r.get("clients", {}).get(cid, {}).get("train_accuracy") for r in rounds]
                   for cid in client_ids}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(title or os.path.basename(output_dir), fontsize=12, fontweight="bold", y=1.01)

    for i, cid in enumerate(client_ids):
        valid = [(x, y) for x, y in zip(xs, client_accs[cid]) if y is not None]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, color=_PALETTE[i % len(_PALETTE)], linewidth=1.5,
                    label=f"client {cid}", alpha=0.85)

    ax.set_xlabel("Round"); ax.set_ylabel("Train Accuracy (%)"); ax.set_title("Per-Client Training Accuracy")
    if len(client_ids) <= 12:
        ax.legend(ncol=min(4, len(client_ids)), fontsize=8, loc="lower right", framealpha=0.7)

    fig.tight_layout()
    return _save(fig, output_dir, "client_accuracy.png")


def plot_data_distribution(output_dir, class_names=None, title=None):
    parts_path  = os.path.join(output_dir, "partitions", "client_partitions.json")
    labels_path = os.path.join(output_dir, "partitions", "labels.npy")
    if not os.path.exists(parts_path) or not os.path.exists(labels_path):
        return None

    with open(parts_path) as f:
        client_indices = json.load(f)
    all_labels  = np.load(labels_path)
    classes     = sorted({int(l) for l in all_labels})
    num_clients = len(client_indices)
    names       = class_names or [str(c) for c in classes]

    counts = np.zeros((num_clients, len(classes)), dtype=int)
    for i, indices in enumerate(client_indices):
        if indices:
            cl = all_labels[np.array(indices, dtype=int)]
            for j, c in enumerate(classes):
                counts[i, j] = (cl == c).sum()

    proportions = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    colors      = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(max(8, num_clients * 0.55), 4.5))
    fig.suptitle(title or f"Data Distribution - {os.path.basename(output_dir)}",
                 fontsize=12, fontweight="bold", y=1.01)

    bottom = np.zeros(num_clients)
    for j, c in enumerate(classes):
        ax.bar(range(num_clients), proportions[:, j], bottom=bottom,
               color=colors[j], label=names[j] if j < len(names) else str(c))
        bottom += proportions[:, j]

    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f"C{i}" for i in range(num_clients)], fontsize=8)
    ax.set_xlabel("Client"); ax.set_ylabel("Class proportion")
    ax.set_title("Class Distribution per Client"); ax.set_ylim(0, 1); ax.grid(False)
    if len(classes) <= 20:
        ax.legend(ncol=min(5, len(classes)), fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.8)

    fig.tight_layout()
    return _save(fig, output_dir, "data_distribution.png")


def plot_class_accuracy_heatmap(output_dir, class_names=None, title=None):
    rounds = _load_rounds(output_dir)
    if not rounds:
        return None

    last       = rounds[-1]
    client_ids = sorted(last.get("clients", {}).keys(), key=lambda x: int(x))
    if not client_ids:
        return None

    all_classes = sorted(
        {cls for cid in client_ids for cls in last["clients"][cid].get("test_class_accuracy", {})},
        key=lambda x: int(x),
    )
    if not all_classes:
        return None

    names  = class_names or all_classes
    matrix = np.full((len(client_ids), len(all_classes)), np.nan)
    for i, cid in enumerate(client_ids):
        tca = last["clients"][cid].get("test_class_accuracy", {})
        for j, cls in enumerate(all_classes):
            v = tca.get(str(cls))
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(7, len(all_classes) * 0.6), max(4, len(client_ids) * 0.45)))
    fig.suptitle(title or f"Per-Class Accuracy (Final Round) - {os.path.basename(output_dir)}",
                 fontsize=11, fontweight="bold", y=1.01)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(all_classes))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(client_ids)));  ax.set_yticklabels([f"client {c}" for c in client_ids], fontsize=8)
    ax.set_xlabel("Class"); ax.set_ylabel("Client")

    for i in range(len(client_ids)):
        for j in range(len(all_classes)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if 20 < v < 80 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")
    fig.tight_layout()
    return _save(fig, output_dir, "class_accuracy_heatmap.png")


def generate_all_plots(output_dir, class_names=None):
    fns = [plot_training_curves, plot_client_accuracy,
           lambda d, **kw: plot_data_distribution(d, class_names=class_names),
           lambda d, **kw: plot_class_accuracy_heatmap(d, class_names=class_names)]
    names = ["training_curves", "client_accuracy", "data_distribution", "class_accuracy_heatmap"]
    generated = []
    for fn, name in zip(fns, names):
        try:
            p = fn(output_dir)
            if p:
                generated.append(p)
        except Exception as e:
            print(f"  [Viz] {name} skipped: {e}")
    if generated:
        print(f"  [Viz] {len(generated)} figures -> {os.path.join(output_dir, 'figures')}/")
    return generated
