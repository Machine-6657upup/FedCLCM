from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.serverrep import FedRep
from FLAlgorithms.users.userfedrepbdpfl import UserFedRepBDPFL
from utils.model_utils import read_data, read_user_data


class ServerFedRepBDPFL(FedRep):
    """FedRep skeleton with BDPFL inter/em submodules only."""

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
        bd_lambda=1.0,
        bd_tau=1.0,
        bd_gamma=1.0,
        bd_use_inter=1,
        bd_use_em=1,
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
        data = read_data(dataset)
        total_users = len(data[0])
        mal_set = set(malclient) if malclient is not None else set()
        for i in range(total_users):
            uid, train, test = read_user_data(i, data, dataset)
            user = UserFedRepBDPFL(
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
                bd_lambda=bd_lambda,
                bd_tau=bd_tau,
                bd_gamma=bd_gamma,
                bd_use_inter=bd_use_inter,
                bd_use_em=bd_use_em,
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:", num_users, " / ", total_users)
        print("Finished creating FedRep + BDPFL-submodule server.")
