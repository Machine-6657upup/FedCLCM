"""
FedRPD server for PFedBA:
- FedRep communication path
- FedRT trimmed aggregation on shared base
- UserFedRPD benign-side purification + distillation
"""
import torch

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serverrep import FedRep
from FLAlgorithms.users.userfedrpd import UserFedRPD
from utils.model_utils import read_data, read_user_data


class ServerFedRPD(FedRep):
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
        aug_strength=0.0,
        adv_eps=0.0,
        adv_num_iter=0,
        purify_beta=1500.0,
        purify_rounds=1,
        distill_gamma=1.0,
        distill_weight=1.0,
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
        data = read_data(dataset)
        total_users = len(data[0])
        mal_set = set(malclient) if malclient is not None else set()
        for i in range(total_users):
            uid, train, test = read_user_data(i, data, dataset)
            user = UserFedRPD(
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
                aug_strength=aug_strength,
                adv_eps=adv_eps,
                adv_num_iter=adv_num_iter,
                purify_beta=purify_beta,
                purify_rounds=purify_rounds,
                distill_gamma=distill_gamma,
                distill_weight=distill_weight,
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", num_users, " / ", total_users)
        self.rt_beta = float(rt_beta)
        print("Finished creating FedRPD server.")

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    def aggregate_parameters(self):
        assert len(self.selected_users) > 0
        for param in self.model.base.parameters():
            param.data = torch.zeros_like(param.data)

        with torch.no_grad():
            for global_param, client_params in zip(
                self.model.base.parameters(),
                zip(*[user.model.base.parameters() for user in self.selected_users])
            ):
                stacked = torch.stack([p.data.clone() for p in client_params], dim=0)
                n = stacked.shape[0]
                k = int(self.rt_beta * n)
                k = min(k, max(0, (n - 1) // 2))
                sorted_params, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_params[k:-k] if k > 0 else sorted_params
                global_param.data.copy_(trimmed.mean(dim=0))
