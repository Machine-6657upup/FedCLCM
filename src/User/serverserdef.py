import random
import time
from User.clientserdef import clientSD
from User.serverbase import Server
from threading import Thread
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

class FedSD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientSD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.learning_rate = args.local_learning_rate
        self.lr_head = args.lr_head
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr= self.learning_rate , momentum=0.9, weight_decay=5e-4)
        self.optimizer_head = torch.optim.SGD(self.global_model.head.parameters(), lr= self.lr_head , momentum=0.9, weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr= 1.)

        self.epsilon = args.adv_eps      # 0.1  # 扰动强度
        self.num_iter = args.adv_num_iter   # PGD迭代次数
        self.alpha = 2 * self.epsilon / self.num_iter  # 每次迭代的步长
        self.plocal_epochs = args.plocal_epochs
        self.defense_epochs = args.defense_epochs
        self.clean_data_num = args.clean_data

        self.num_clients = args.num_clients
        self.num_adv_clients = args.num_adv_clients


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            # 1:user to test the backdoor rate
            for client in self.selected_clients:
                client.train()

            ASR, attack_auc = self.test_backdoor_metrics()
            print(f"ASR : {ASR}")
            print(f"attack_auc : {attack_auc}")

            self.train_server_head()

            self.receive_models()

            self.aggregate_parameters()
            
            self.backdoor_defense()

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
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        
    def send_models(self, round):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            # if(round == 0):
            #     client.set_parameters(self.global_model)
            # else:
            #     client.set_parameters(self.global_model.base)
            client.set_parameters(self.global_model, round)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    def train_server_head(self):
        cleanloader = self.load_clean_data()
        self.global_model.train()
        for param in self.global_model.base.parameters():
            param.requires_grad = False
        for param in self.global_model.head.parameters():
            param.requires_grad = True
        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(cleanloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                loss = self.loss(output, y)
                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_head.step()



    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model.base = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.base.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)



    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.base.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

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
                # self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def backdoor_defense(self):
        cleanloader = self.load_clean_data()
        for param in self.global_model.parameters():
            param.requires_grad = True
        for _ in range(self.defense_epochs):
            for i, (x, y) in enumerate(cleanloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x_orig = x.clone().detach()
                delta = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
                adv_x = x_orig + delta
                adv_x = torch.clamp(adv_x, -1, 1)

                self.global_model.eval()

                for _ in range(self.num_iter):
                    adv_x.requires_grad_(True)
                    
                    # 前向计算
                    with torch.enable_grad():
                        outputs = self.global_model(adv_x)
                        loss = F.cross_entropy(outputs, y)
                    
                    # 计算输入梯度
                    grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
                    
                    # 更新对抗扰动
                    adv_x = adv_x.detach() + self.alpha * grad.sign()
                    delta = adv_x - x_orig
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)  # 投影到epsilon邻域
                    # adv_x = x_orig + delta
                    adv_x = torch.clamp(x_orig + delta, -1, 1).detach()  # 这里改到了-1到1


                self.global_model.train()

                output = self.global_model(adv_x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def load_clean_data(self):
        clean_data_dir = os.path.join('./dataset', self.dataset, 'test/')
        clean_file = clean_data_dir + 'server_clean.npz'
        with open(clean_file, 'rb') as f:
            clean_data = np.load(f, allow_pickle=True)['data'].tolist()
        #X_clean = torch.Tensor(clean_data['x'][:self.clean_data_num]).type(torch.float32)
        #y_clean = torch.Tensor(clean_data['y'][:self.clean_data_num]).type(torch.int64)
        X_clean = torch.Tensor(clean_data['x']).type(torch.float32)
        y_clean = torch.Tensor(clean_data['y']).type(torch.int64)
        clean_data = [(x, y) for x, y in zip(X_clean, y_clean)]
        category_dict = {i: [] for i in range(self.num_classes)}
        for x, y in clean_data:
            category_dict[y.item()].append((x, y))
        clean_data = []
        for i in range(self.num_classes):
            samples = category_dict[i]
            selected = samples[:self.clean_data_num//self.num_classes]
            clean_data.extend(selected)

        return DataLoader(clean_data, self.batch_size, drop_last=True, shuffle=True)


    def test_backdoor_metrics(self):
        total_ASR = 0.0
        total_auc = 0.0
        num_clients = 10  # 假设共10个客户端

        for id in range(1, num_clients):
            backdoor_data_dir = os.path.join('./dataset', self.clients[id].dataset, 'test/')
            backdoor_file = os.path.join(backdoor_data_dir, f'{id}_backdoored.npz') 

            with open(backdoor_file, 'rb') as f:
                backdoor_data = np.load(f, allow_pickle=True)['data'].tolist()

            X_backdoor = torch.Tensor(backdoor_data['x']).type(torch.float32)
            y_backdoor = torch.Tensor(backdoor_data['y']).type(torch.int64)

            backdoor_data = [(x, y) for x, y in zip(X_backdoor, y_backdoor)]
            backdoorloaderfull = DataLoader(backdoor_data, self.clients[id].batch_size, drop_last=False, shuffle=True)

            # self.model = self.load_model('model')
            # self.model.to(self.device)
            self.clients[id].model.eval()

            test_acc = 0
            test_num = 0
            y_prob = []
            y_true = []
            
            with torch.no_grad():
                for i, (x, y) in enumerate(backdoorloaderfull):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.clients[id].device)
                    else:
                        x = x.to(self.clients[id].device)
                    y = y.to(self.clients[id].device)
                    #print(y)
                    output = self.clients[id].model(x)
                    #print(output)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    # if(i == 0):
                    #     print(f"backdoor dataset label:{y}")
                    #     print(f"backdoor data predict:{torch.argmax(output, dim=1)}")
                    test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.clients[id].num_classes
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    y_true.append(lb)


            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            ASR = test_acc*1.0 / test_num
            print(f"id {id} ASR: {ASR}")

            total_ASR += ASR  # 累加ASR
            total_auc += auc

        avg_ASR = total_ASR / 9  # 计算平均ASR
        avg_auc = total_auc / 9
            
        return avg_ASR, avg_auc