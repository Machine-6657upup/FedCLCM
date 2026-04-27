"""
FedCLCM server for PFedBA: FedRep communication + trimmed-mean on shared base,
optional channel mask and cosine gating (ported from root User/serverCLCM.py).
"""
import copy

import numpy as np
import torch

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serverrep import FedRep
from FLAlgorithms.users.userclcm import UserCLCM
from utils.model_utils import read_data, read_user_data


class ServerFedCLCM(FedRep):
    """FedRep 训练循环 + FedCLCM 聚合；继承 FedRep 以获得 FedAvg.train()。"""
    user_class = UserCLCM

    def __init__(
        self,
        device,
        dataset,
        algorithm,
        model,
        batch_size,
        learning_rate,
        beta,
        lamda,
        num_glob_iters,
        local_epochs,
        optimizer,
        num_users,
        times,
        fo,
        current_time,
        malnum,
        malclient,
        poisonratio,
        poison_label,
        attack_method,
        per_epoch,
        defense,
        lr_head=None,
        plocal_epochs=1,
        rt_beta=0.2,
        mask_tau=2.0,
        mask_alpha=0.3,
        enable_channel_mask=True,
        lambda_cl=0.5,
        aug_strength=0.1,
        adv_eps=0.0,
        adv_num_iter=0,
        cosine_gate=False,
        cosine_gate_threshold=0.3,
        cosine_gate_alpha=0.5,
        trim_high_layers=None,
        trim_beta_high=None,
        trim_beta_low=None,
    ):
        # 只初始化 Server，不跑 FedAvg.__init__（避免先建 UserAVG）；用户由本类建为 UserCLCM。
        Server.__init__(
            self,
            device,
            dataset,
            algorithm,
            model,
            batch_size,
            learning_rate,
            beta,
            lamda,
            num_glob_iters,
            local_epochs,
            optimizer,
            num_users,
            times,
            fo,
            current_time,
            malnum,
            malclient,
            poisonratio,
            poison_label,
            attack_method,
            per_epoch,
            defense,
        )
        self.users = []
        self.total_train_samples = 0
        data = read_data(dataset)
        total_users = len(data[0])
        mal_set = set(malclient) if malclient is not None else set()
        user_class = getattr(self, "user_class", UserCLCM)
        extra_user_kwargs = getattr(self, "extra_user_kwargs", {})
        for i in range(total_users):
            uid, train, test = read_user_data(i, data, dataset)
            user = user_class(
                device,
                uid,
                train,
                test,
                model,
                dataset,
                batch_size,
                learning_rate,
                beta,
                lamda,
                local_epochs,
                lr_head=lr_head,
                plocal_epochs=plocal_epochs,
                malclient_ids=mal_set,
                lambda_cl=lambda_cl,
                aug_strength=aug_strength,
                adv_eps=adv_eps,
                adv_num_iter=adv_num_iter,
                **extra_user_kwargs,
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", num_users, " / ", total_users)

        self.rt_beta = float(rt_beta)
        self.mask_tau = float(mask_tau)
        self.mask_alpha = float(mask_alpha)
        self.enable_channel_mask = bool(enable_channel_mask)
        self.cosine_gate = bool(cosine_gate)
        self.cosine_gate_threshold = float(cosine_gate_threshold)
        self.cosine_gate_alpha = float(cosine_gate_alpha)
        if trim_high_layers is None:
            trim_high_layers = []
        elif isinstance(trim_high_layers, str):
            trim_high_layers = [s.strip() for s in trim_high_layers.split(",") if s.strip()]
        self.trim_high_layers = trim_high_layers
        self.trim_beta_high = trim_beta_high
        self.trim_beta_low = trim_beta_low
        self.channel_mask = {}
        self.prev_global_model = None
        print("Finished creating FedCLCM (PFedBA) server.")

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    def get_current_global_base(self):
        return self.model.base

    def pre_client_train_hook(self, glob_iter):
        # Snapshot the current shared base before local training so all
        # round-level update statistics use the correct reference point.
        self.prev_global_model = copy.deepcopy(self.get_current_global_base())

    def pre_aggregate_hook(self, glob_iter):
        if self.enable_channel_mask and glob_iter > 0:
            self.update_channel_mask()

    def compute_updates(self):
        if self.prev_global_model is None:
            return None
        updates = []
        prev = self.prev_global_model
        for user in self.selected_users:
            upd = {}
            for (name, prev_p), (_, client_p) in zip(prev.named_parameters(), user.model.base.named_parameters()):
                upd[name] = client_p.data - prev_p.data
            updates.append(upd)
        return updates

    def compute_client_weights(self):
        if (not self.cosine_gate) or self.prev_global_model is None:
            return None
        updates = self.compute_updates()
        if updates is None:
            return None
        vecs = []
        for update in updates:
            flat = []
            for _, t in update.items():
                flat.append(t.detach().reshape(-1))
            if not flat:
                return None
            vecs.append(torch.cat(flat))
        if not vecs:
            return None
        mat = torch.stack(vecs, dim=0)
        mean_vec = mat.mean(dim=0)
        eps = 1e-12
        mean_norm = torch.norm(mean_vec) + eps
        mat_norm = torch.norm(mat, dim=1) + eps
        cos = (mat @ mean_vec) / (mat_norm * mean_norm)
        weights = torch.ones_like(cos, device=cos.device)
        low_mask = cos < self.cosine_gate_threshold
        weights[low_mask] = self.cosine_gate_alpha
        return weights

    def _get_layer_beta(self, name):
        if self.trim_beta_high is None and self.trim_beta_low is None:
            return self.rt_beta
        is_high = any(name.startswith(prefix) for prefix in self.trim_high_layers)
        if is_high:
            return self.trim_beta_high if self.trim_beta_high is not None else self.rt_beta
        return self.trim_beta_low if self.trim_beta_low is not None else self.rt_beta

    def update_channel_mask(self):
        updates = self.compute_updates()
        if updates is None:
            return
        prev = self.prev_global_model
        for name, param in prev.named_parameters():
            if "weight" not in name or len(param.shape) < 2:
                continue
            num_channels = param.shape[0]
            layer_updates = [u[name] for u in updates]
            variances = []
            for c in range(num_channels):
                channel_updates = torch.stack([upd[c].flatten() for upd in layer_updates])
                variances.append(channel_updates.var(dim=0).mean().item())
            variances = np.array(variances)
            threshold = float(np.median(variances)) * self.mask_tau
            mask = torch.ones(num_channels, device=param.device, dtype=param.dtype)
            for c, v in enumerate(variances):
                if v > threshold:
                    mask[c] = self.mask_alpha
            self.channel_mask[name] = mask
            num_suspicious = (mask < 1.0).sum().item()
            if num_suspicious > 0:
                print(f"[FedCLCM] Layer {name}: {num_suspicious}/{num_channels} channels masked")

    def aggregate_parameters(self):
        # select_users() 在 num_users < 全体用户数时返回 np.ndarray，不能写 `if self.selected_users`
        assert len(self.selected_users) > 0
        users_to_agg = self.selected_users
        client_weights = self.compute_client_weights()
        client_param_dicts = []
        for u in users_to_agg:
            client_param_dicts.append(dict(u.model.base.named_parameters()))

        with torch.no_grad():
            for name, global_param in self.model.base.named_parameters():
                if name not in client_param_dicts[0]:
                    continue
                stacked = torch.stack([d[name].data.clone() for d in client_param_dicts], dim=0)
                if client_weights is not None:
                    w_shape = [client_weights.shape[0]] + [1] * (stacked.dim() - 1)
                    stacked = stacked * client_weights.view(*w_shape)
                if self.enable_channel_mask and name in self.channel_mask:
                    mask = self.channel_mask[name]
                    mask_shape = [1, -1] + [1] * (len(stacked.shape) - 2)
                    stacked = stacked * mask.view(*mask_shape)
                n = stacked.shape[0]
                beta = self._get_layer_beta(name)
                k = int(beta * n)
                k = min(k, max(0, (n - 1) // 2))
                sorted_params, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_params[k:-k] if k > 0 else sorted_params
                global_param.data.copy_(trimmed.mean(dim=0))
