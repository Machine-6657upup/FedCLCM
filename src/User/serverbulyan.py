import copy

import numpy as np
import torch

from User.serveravg import FedAvg


class FedBulyan(FedAvg):
    """Bulyan-style baseline with safe fallback.

    Exact Bulyan requires enough uploaded clients for the estimated fault count
    (typically n >= 4f + 3). If the current round violates that condition, this
    falls back to coordinate-wise trimmed mean instead of returning an unstable
    selection.
    """

    def __init__(self, args, times):
        super().__init__(args, times)
        self.trim_ratio = getattr(args, "trim_ratio", 0.1)

    def _estimate_f(self, num_clients):
        selected_adv = sum(1 for cid in getattr(self, "uploaded_ids", []) if cid < self.num_adv_clients)
        fallback = int(self.trim_ratio * num_clients)
        f = selected_adv if selected_adv > 0 else fallback
        return max(0, min(f, (num_clients - 1) // 2))

    @staticmethod
    def _flatten_model(model):
        return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])

    @staticmethod
    def _load_flat_params(model, flat_params):
        offset = 0
        with torch.no_grad():
            for param in model.parameters():
                size = param.numel()
                value = flat_params[offset : offset + size].reshape(param.shape).to(param.device)
                param.data.copy_(value)
                offset += size

    def _trimmed_mean_flat(self, flat_stack, f):
        if f > 0:
            sorted_params, _ = torch.sort(flat_stack, dim=0)
            trimmed = sorted_params[f:-f]
        else:
            trimmed = flat_stack
        return trimmed.mean(dim=0)

    def _krum_scores(self, flat_stack, f):
        arr = flat_stack.numpy()
        n = arr.shape[0]
        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i):
                d = np.linalg.norm(arr[i] - arr[j])
                distances[i, j] = d
                distances[j, i] = d

        neighbor_count = max(1, min(n - 1, n - f - 2))
        scores = []
        for i in range(n):
            nearest = np.sort(distances[i])[1 : neighbor_count + 1]
            scores.append(float(np.sum(nearest)))
        return np.asarray(scores)

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        num_clients = len(self.uploaded_models)
        f = self._estimate_f(num_clients)
        flat_stack = torch.stack([self._flatten_model(model) for model in self.uploaded_models], dim=0)

        if f == 0:
            aggregated = flat_stack.mean(dim=0)
        elif num_clients < 4 * f + 3:
            aggregated = self._trimmed_mean_flat(flat_stack, f)
        else:
            theta = num_clients - 2 * f
            scores = self._krum_scores(flat_stack, f)
            selected = np.argsort(scores)[:theta]
            selected_stack = flat_stack[selected]
            aggregated = self._trimmed_mean_flat(selected_stack, f)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        self._load_flat_params(self.global_model, aggregated)
