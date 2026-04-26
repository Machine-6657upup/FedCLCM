import torch

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.users.userrep import UserRep
from utils.model_utils import read_data, read_user_data


class FedRep(FedAvg):
    """FedRep server with shared-base aggregation only."""

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
    ):
        # Keep FedAvg.train() behavior, but avoid FedAvg.__init__ because it
        # eagerly builds UserAVG for all clients before FedRep rebuilds UserRep.
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
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            uid, train, test = read_user_data(i, data, dataset)
            user = UserRep(
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
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:", num_users, " / ", total_users)
        print("Finished creating FedRep server.")

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    def aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for param in self.model.base.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            ratio = user.train_samples / total_train
            for server_param, user_param in zip(self.model.base.parameters(), user.model.base.parameters()):
                server_param.data += user_param.data.clone() * ratio
