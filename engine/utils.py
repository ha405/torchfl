import torch
import torch.nn as nn
import os
import glob


def evaluate_model(model, cache, config, class_names=None, title="Eval", log_file=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    nc = config.num_classes
    class_correct = [0.0] * nc
    class_total   = [0.0] * nc
    correct = total = running_loss = n_batches = 0

    if log_file:
        log_file.write(f"\n--- {title} ---\n")

    with torch.no_grad():
        for inputs, labels in cache.iterate_batches(config.batch_size):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            mask = labels < nc
            if not mask.any():
                continue
            inputs, labels = inputs[mask], labels[mask]
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            n_batches    += 1
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()
            is_correct    = predicted == labels
            for c in range(nc):
                class_correct[c] += torch.bincount(labels[is_correct], minlength=nc)[c].item()
                class_total[c]   += torch.bincount(labels,             minlength=nc)[c].item()

    overall_acc = 100 * correct / total if total > 0 else 0.0
    avg_loss    = running_loss / n_batches if n_batches > 0 else 0.0
    class_acc   = {c: (100 * class_correct[c] / class_total[c] if class_total[c] > 0 else None)
                   for c in range(nc)}

    if log_file:
        log_file.write(f"Overall: {overall_acc:.2f}% | Loss: {avg_loss:.4f}\n")
        for c in range(nc):
            name = class_names[c] if class_names else str(c)
            if class_acc[c] is not None:
                log_file.write(f"  {name}: {class_acc[c]:.2f}% ({int(class_correct[c])}/{int(class_total[c])})\n")
            else:
                log_file.write(f"  {name}: N/A\n")
        log_file.write("-" * 30 + "\n")
        log_file.flush()

    return overall_acc, avg_loss, class_acc


def load_latest_checkpoint(global_model, config):
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_round_*.pt"))
    if not files:
        return 0
    latest = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    print(f"  [Resume] Loading {latest}")
    ckpt = torch.load(latest, map_location=config.device)
    global_model.load_state_dict(ckpt["model_state_dict"])
    print(f"  [Resume] Resuming from round {ckpt['round'] + 1}")
    return ckpt["round"]


def save_checkpoint(global_model, round_num, config, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pt")
    torch.save({"round": round_num, "model_state_dict": global_model.state_dict()}, path)
    print(f"  [Checkpoint] {path}")


def save_local_model(model, round_num, client_idx, config):
    d = os.path.join(config.output_dir, "checkpoints", f"round_{round_num + 1}")
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, f"client_{client_idx}_model.pt"))
