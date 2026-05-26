import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import List

from .client import FederatedClient
from engine.utils import save_local_model
from engine.data_cache import EvaluationCache


class FederatedServer:
    def __init__(self, global_model, config, class_names, evaluation_cache: EvaluationCache):
        self.global_model     = global_model
        self.config           = config
        self.class_names      = class_names
        self.device           = config.device
        self.evaluation_cache = evaluation_cache

        self.client_streams = (
            [torch.cuda.Stream() for _ in range(config.num_clients)]
            if "cuda" in str(config.device) else []
        )

        # lazily initialized on first SCAFFOLD round
        self._scaffold_c  = None
        self._scaffold_ci = {}

    def _clone_global(self):
        from models import get_model
        m = get_model(self.config)
        m.load_state_dict(self.global_model.state_dict())
        return m.to(self.device)

    def _select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        frac    = getattr(self.config, "fraction_fit", 1.0)
        minimum = max(getattr(self.config, "min_fit_clients", 1), 1)
        k       = max(minimum, round(frac * len(clients)))
        k       = min(k, len(clients))
        return random.sample(clients, k)

    def _scaffold_state_for(self, client):
        if self._scaffold_c is None:
            self._scaffold_c = {
                k: torch.zeros_like(v)
                for k, v in self.global_model.state_dict().items()
                if v.is_floating_point()
            }
        cid = client.client_id
        if cid not in self._scaffold_ci:
            self._scaffold_ci[cid] = {
                k: torch.zeros_like(v)
                for k, v in self.global_model.state_dict().items()
                if v.is_floating_point()
            }
        return self._scaffold_c, self._scaffold_ci[cid]

    def _aggregate_fedavg(self, updates):
        total       = sum(u["dataset_size"] for u in updates)
        state_dicts = [u["model"].state_dict() for u in updates]
        weights     = [u["dataset_size"] / total for u in updates]
        new_state   = {}
        for key in state_dicts[0]:
            acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc.add_(sd[key].float(), alpha=w)
            new_state[key] = acc.to(self.device)
        self.global_model.load_state_dict(new_state)

    def _aggregate_fednova(self, updates):
        total     = sum(u["dataset_size"] for u in updates)
        tau_eff   = sum(u["dataset_size"] / total * u["num_steps"] for u in updates)
        global_sd = {k: v.clone().float() for k, v in self.global_model.state_dict().items()}
        delta_avg = {k: torch.zeros_like(v) for k, v in global_sd.items()}

        for u in updates:
            w     = u["dataset_size"] / total
            steps = max(u["num_steps"], 1)
            sd    = u["model"].state_dict()
            for key in global_sd:
                delta_avg[key].add_((global_sd[key] - sd[key].float()) / steps, alpha=w)

        self.global_model.load_state_dict(
            {key: (global_sd[key] - tau_eff * delta_avg[key]).to(self.device)
             for key in global_sd}
        )

    def _aggregate_scaffold(self, updates, selected_clients):
        self._aggregate_fedavg(updates)

        n = len(updates)
        for u, client in zip(updates, selected_clients):
            delta_c = u.get("delta_c")
            if delta_c is None:
                continue
            cid = client.client_id
            for k in delta_c:
                self._scaffold_ci[cid][k] = self._scaffold_ci[cid][k] + delta_c[k]
                self._scaffold_c[k]       = self._scaffold_c[k] + delta_c[k] / n

    def orchestrate_round(self, round_num: int, clients: List[FederatedClient],
                          log_file=None):
        algo     = getattr(self.config, "fl_algorithm", "fedavg").lower()
        selected = self._select_clients(clients)
        n        = len(selected)

        client_train_metrics = {}
        client_test_metrics  = {}
        updates: list        = [None] * n

        def _train(i, client):
            stream = self.client_streams[client.client_id] \
                if self.client_streams and client.client_id < len(self.client_streams) else None
            ctx = torch.cuda.stream(stream) if stream else nullcontext()
            with ctx:
                model_copy = self._clone_global()
                extra      = {}
                if algo == "scaffold":
                    c, ci = self._scaffold_state_for(client)
                    extra = {"scaffold_c": c, "scaffold_ci": ci}
                result   = client.train(model_copy, round_num, **extra)
                test_acc = client.evaluate_on_test(result["model"], self.evaluation_cache)
                if stream:
                    stream.synchronize()
            save_local_model(result["model"], round_num, client.client_id, self.config)
            return i, result, test_acc

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_train, i, c): i for i, c in enumerate(selected)}
            for i, result, test_acc in [f.result() for f in as_completed(futures)]:
                updates[i]                                  = result
                client_train_metrics[selected[i].client_id] = result["metrics"]
                client_test_metrics[selected[i].client_id]  = test_acc

        if algo == "fednova":
            self._aggregate_fednova(updates)
        elif algo == "scaffold":
            self._aggregate_scaffold(updates, selected)
        else:
            self._aggregate_fedavg(updates)

        return client_train_metrics, client_test_metrics, len(selected)
