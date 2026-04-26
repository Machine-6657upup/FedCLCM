import copy

import numpy as np
import torch

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serverrep import FedRep
from FLAlgorithms.users.userfedreppflalp import UserFedRepPFLALP
from utils.model_utils import read_data, read_user_data


class ServerFedRepPFLALP(FedRep):
    """FedRep skeleton with PFL-ALP ablation modules only."""

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
        alp_use_cluster=1,
        alp_use_purify=1,
        purify_beta=1500.0,
        purify_rounds=1,
        cluster_max_k=4,
    ):
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
        self.alp_use_cluster = bool(alp_use_cluster)
        self.alp_use_purify = bool(alp_use_purify)
        self.cluster_max_k = max(int(cluster_max_k), 2)
        self.prev_global_model = None

        data = read_data(dataset)
        total_users = len(data[0])
        mal_set = set(malclient) if malclient is not None else set()
        for i in range(total_users):
            uid, train, test = read_user_data(i, data, dataset)
            user = UserFedRepPFLALP(
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
                alp_use_purify=alp_use_purify,
                purify_beta=purify_beta,
                purify_rounds=purify_rounds,
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", num_users, " / ", total_users)
        print("Finished creating FedRep + PFL-ALP-submodule server.")

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    def pre_one_step_eval_hook(self):
        # PFL-ALP keeps a personalized model for inference / purification
        # guidance, but the generic "k local SGD" metric in this codebase is
        # defined as fine-tuning from the current global model.
        return

    def pre_client_train_hook(self, glob_iter):
        # Paper logic: the next-round local update warm-starts from the global
        # model. Personalized models stay on the client for inference and as
        # purification teachers, not as the communication-model initializer.
        return

    def pre_aggregate_hook(self, glob_iter):
        if self.alp_use_cluster:
            self.prev_global_model = copy.deepcopy(self.model)

    def _flatten_base_update(self, user):
        flat = []
        for prev_param, user_param in zip(self.prev_global_model.base.parameters(), user.model.base.parameters()):
            flat.append((user_param.data - prev_param.data).reshape(-1).detach().cpu())
        return torch.cat(flat).numpy()

    def _build_score_matrix(self, selected_users):
        updates = [self._flatten_base_update(user) for user in selected_users]
        score = np.zeros((len(updates), len(updates)), dtype=np.float64)
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                a = updates[i]
                b = updates[j]
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                dist = 1.0 - float(np.dot(a, b) / denom)
                score[i, j] = dist
                score[j, i] = dist
        return score

    def _kmeans(self, data, k, max_iter=30):
        n = data.shape[0]
        centers = data[:k].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centers[idx] = data[mask].mean(axis=0)
        return labels

    def _silhouette_score(self, data, labels):
        n = data.shape[0]
        unique = np.unique(labels)
        if len(unique) <= 1 or len(unique) >= n:
            return -1.0
        pairwise = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        scores = []
        for i in range(n):
            own = labels[i]
            own_mask = labels == own
            own_count = int(np.sum(own_mask))
            if own_count <= 1:
                scores.append(0.0)
                continue
            a = pairwise[i, own_mask].sum() / (own_count - 1)
            b = None
            for other in unique:
                if other == own:
                    continue
                other_mask = labels == other
                dist = pairwise[i, other_mask].mean()
                if b is None or dist < b:
                    b = dist
            scores.append(0.0 if b is None else (b - a) / max(a, b, 1e-12))
        return float(np.mean(scores))

    def _dynamic_cluster(self, selected_users):
        n = len(selected_users)
        if n <= 2:
            return np.zeros(n, dtype=np.int64)
        data = self._build_score_matrix(selected_users)
        best_labels = np.zeros(n, dtype=np.int64)
        best_score = -1.0
        max_k = min(self.cluster_max_k, n - 1)
        for k in range(2, max_k + 1):
            labels = self._kmeans(data, k)
            if len(np.unique(labels)) < 2:
                continue
            score = self._silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels.copy()
        if best_score <= 0:
            return np.zeros(n, dtype=np.int64)
        return best_labels

    def _build_representative_model(self, cluster_users):
        representative = copy.deepcopy(self.model)
        total = float(sum(user.train_samples for user in cluster_users))
        for rep_param in representative.base.parameters():
            rep_param.data = torch.zeros_like(rep_param.data)
        for user in cluster_users:
            weight = user.train_samples / total if total > 0 else (1.0 / len(cluster_users))
            for rep_param, user_param in zip(representative.base.parameters(), user.model.base.parameters()):
                rep_param.data += user_param.data.clone() * weight
        return representative

    def post_aggregate_hook(self, glob_iter):
        if len(self.selected_users) == 0:
            return

        if self.alp_use_cluster and self.prev_global_model is not None:
            labels = self._dynamic_cluster(self.selected_users)
            representatives = {}
            for cluster_id in np.unique(labels):
                cluster_users = [user for user, lab in zip(self.selected_users, labels) if lab == cluster_id]
                representatives[int(cluster_id)] = self._build_representative_model(cluster_users)
        else:
            labels = np.zeros(len(self.selected_users), dtype=np.int64)
            representatives = {0: copy.deepcopy(self.model)}

        print("[FedRepPFLALP] cluster labels:", labels.tolist())
        for user, cluster_id in zip(self.selected_users, labels):
            user.personalize_with_representative(representatives[int(cluster_id)])
