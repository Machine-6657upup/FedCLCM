import copy

import torch

from User.serveravg import FedAvg


class FedTrimmed(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.trim_ratio = getattr(args, "trim_ratio", 0.1)

    def _trim_count(self, num_clients):
        selected_adv = sum(1 for cid in getattr(self, "uploaded_ids", []) if cid < self.num_adv_clients)
        fallback = int(self.trim_ratio * num_clients)
        f = selected_adv if selected_adv > 0 else fallback
        return max(0, min(f, (num_clients - 1) // 2))

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for global_param, client_params in zip(
            self.global_model.parameters(),
            zip(*[model.parameters() for model in self.uploaded_models]),
        ):
            stacked_params = torch.stack([p.data.clone() for p in client_params], dim=0)
            num_clients = stacked_params.shape[0]
            k = self._trim_count(num_clients)

            if k > 0:
                sorted_params, _ = torch.sort(stacked_params, dim=0)
                trimmed_params = sorted_params[k:-k]
            else:
                trimmed_params = stacked_params

            global_param.data.copy_(trimmed_params.mean(dim=0))
