import copy

import numpy as np
import torch

from User.serveravg import FedAvg


class FedMK(FedAvg):
    """Multi-Krum-style robust aggregation.

    The fault estimate follows the PFedBA reference implementation: for static
    experiments, malicious clients are the first ``num_adv_clients`` ids, so the
    server can count how many of them actually uploaded this round.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

    def _estimate_f(self, num_clients):
        selected_adv = sum(1 for cid in getattr(self, "uploaded_ids", []) if cid < self.num_adv_clients)
        fallback = int(0.1 * num_clients)
        f = selected_adv if selected_adv > 0 else fallback
        return max(0, min(f, max(0, (num_clients - 3) // 2)))

    @staticmethod
    def _flatten_model(model):
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        num_clients = len(self.uploaded_models)
        f = self._estimate_f(num_clients)
        if num_clients == 1 or f == 0:
            return super().aggregate_parameters()

        flat_params = [self._flatten_model(model) for model in self.uploaded_models]
        distances = np.zeros((num_clients, num_clients), dtype=np.float64)
        for i in range(num_clients):
            for j in range(i):
                d = np.linalg.norm(flat_params[i] - flat_params[j])
                distances[i, j] = d
                distances[j, i] = d

        neighbor_count = max(1, min(num_clients - 1, num_clients - f - 2))
        scores = []
        for i in range(num_clients):
            nearest = np.sort(distances[i])[1 : neighbor_count + 1]
            scores.append(float(np.sum(nearest)))

        select_count = max(1, min(num_clients, num_clients - f))
        selected_indices = np.argsort(scores)[:select_count]

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        with torch.no_grad():
            for idx in selected_indices:
                for global_param, local_param in zip(
                    self.global_model.parameters(),
                    self.uploaded_models[idx].parameters(),
                ):
                    global_param.data.add_(local_param.data.to(global_param.device) / len(selected_indices))
