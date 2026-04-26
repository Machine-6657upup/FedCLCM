import random
import time
from User.clientrepclidef import clientCliDef
from User.serverbase import Server
from threading import Thread
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import copy

class FedRepTrim(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientCliDef)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.beta = args.rt_beta


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            # 1:user to test the backdoor rate
            for client in self.selected_clients:
                client.train()

            #if(i%3 == 0):
            ASR, attack_auc = self.test_backdoor_metrics()
            print(f"ASR : {ASR}")
            print(f"attack_auc : {attack_auc}")

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCliDef)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for global_param, client_params in zip(
            self.global_model.parameters(),
            zip(*[model.parameters() for model in self.uploaded_models])
        ):
        # 将各客户端该层参数堆叠成张量 [num_clients, layer_shape...]
            stacked_params = torch.stack([p.data.clone() for p in client_params], dim=0)
            num_clients = stacked_params.shape[0]
            k = int(self.beta * num_clients)
            
            # 沿客户端维度排序
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            # 执行截断（去除前k个和后k个）
            trimmed_params = sorted_params[k:-k] if k > 0 else sorted_params
            # 计算均值并更新全局参数
            global_param.data = trimmed_params.mean(dim=0)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
