import os
import json
import time
import csv
import platform
import subprocess


class MetricsTracker:
    def __init__(self, output_dir: str):
        self.output_dir    = output_dir
        self.rounds        = []
        self._metrics_path = os.path.join(output_dir, "metrics.json")
        self._csv_path     = os.path.join(output_dir, "metrics.csv")

    def load(self):
        if os.path.exists(self._metrics_path):
            try:
                with open(self._metrics_path) as f:
                    self.rounds = json.load(f).get("rounds", [])
                print(f"  [Metrics] Resumed {len(self.rounds)} rounds from {self._metrics_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [Metrics] Could not load: {e}")
                self.rounds = []

    def log_round(self, round_num: int, global_acc: float, global_loss: float,
                  global_class_acc: dict = None, client_train_metrics: dict = None,
                  client_test_metrics: dict = None, round_time: float = None):
        entry = {
            "round":           round_num + 1,
            "global_accuracy": round(global_acc, 4),
            "global_loss":     round(global_loss, 6),
            "timestamp":       time.time(),
        }
        if round_time is not None:
            entry["round_time_seconds"] = round(round_time, 1)
        if global_class_acc:
            entry["global_class_accuracy"] = {
                str(k): round(v, 4) for k, v in global_class_acc.items() if v is not None
            }
        if client_train_metrics:
            entry["clients"] = {}
            for cid, m in client_train_metrics.items():
                entry["clients"][str(cid)] = {
                    "train_loss":     round(m.get("loss", 0), 6),
                    "train_accuracy": round(m.get("accuracy", 0), 4),
                }
        if client_test_metrics:
            entry.setdefault("clients", {})
            for cid, class_acc in client_test_metrics.items():
                entry["clients"].setdefault(str(cid), {})["test_class_accuracy"] = {
                    str(k): v for k, v in class_acc.items()
                }
        self.rounds.append(entry)

    def save(self):
        with open(self._metrics_path, "w") as f:
            json.dump({"rounds": self.rounds}, f, indent=2)

    def save_csv(self):
        if not self.rounds:
            return
        client_ids = []
        for r in self.rounds:
            if "clients" in r:
                client_ids = sorted(r["clients"].keys(), key=lambda x: int(x))
                break
        header = ["round", "global_accuracy", "global_loss"]
        for cid in client_ids:
            header += [f"client_{cid}_train_loss", f"client_{cid}_train_accuracy"]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in self.rounds:
                row = [r["round"], r["global_accuracy"], r["global_loss"]]
                for cid in client_ids:
                    d = r.get("clients", {}).get(cid, {})
                    row += [d.get("train_loss", ""), d.get("train_accuracy", "")]
                writer.writerow(row)


class RunInfo:
    def __init__(self, output_dir: str, config):
        self.path = os.path.join(output_dir, "run_info.json")
        self._start = time.time()
        self.data = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None, "duration_seconds": None, "final_accuracy": None,
            "system": self._get_system_info(),
            "config": config.to_dict() if hasattr(config, "to_dict") else str(config),
        }
        self.save()

    def _get_system_info(self):
        import torch
        info = {
            "python_version":  platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform":        platform.platform(),
            "cpu":             platform.processor() or "unknown",
        }
        if torch.cuda.is_available():
            info["gpu"]           = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        else:
            info["gpu"] = None
        try:
            info["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            info["git_commit"] = None
        return info

    def finish(self, final_accuracy: float = None):
        self.data["end_time"]         = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["duration_seconds"] = round(time.time() - self._start, 1)
        if final_accuracy is not None:
            self.data["final_accuracy"] = round(final_accuracy, 4)
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
