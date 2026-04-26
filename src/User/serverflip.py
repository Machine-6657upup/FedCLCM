# serverflip.py
import time
import torch
import numpy as np
from User.serverbase import Server
from User.clientflip import clientFLIP
import copy

class FedFLIP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientFLIP)
        self.Budget = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            # 客户端本地训练
            for client in self.selected_clients:
                client.train()
            
            ASR, attack_auc = self.test_backdoor_metrics()
            print(f"ASR : {ASR}")
            print(f"attack_auc : {attack_auc}")

            # 模型聚合
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()

    def aggregate_parameters(self):
        # 实现鲁棒聚合（示例使用Trimmed Mean）
        updates = []
        for client in self.selected_clients:
            updates.append((client.train_samples, copy.deepcopy(client.model.state_dict())))
        
        # Trimmed Mean聚合
        sorted_updates = sorted(updates, key=lambda x: self.cosine_sim(x[1]))
        trimmed = sorted_updates[self.num_adv_clients:-self.num_adv_clients]
        total_size = sum([num for num, _ in trimmed])
        
        global_dict = {}
        for name, param in self.global_model.state_dict().items():
            global_dict[name] = torch.zeros_like(param, dtype=torch.float32)
            for num, model in trimmed:
                global_dict[name] += model[name] * (num / total_size)
        
        self.global_model.load_state_dict(global_dict)

    def cosine_sim(self, model_dict):
        # 计算模型相似度（用于鲁棒聚合）
        global_params = self.global_model.state_dict()
        sim = 0
        for key in global_params:
            a = global_params[key].flatten().float()  # 转换为Float类型
            b = model_dict[key].flatten().float()     # 转换为Float类型
            sim += torch.dot(a, b) / (torch.norm(a)*torch.norm(b) + 1e-8)
        return sim.item()