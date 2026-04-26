import time
import numpy as np
from User.clientproto import clientProto
from User.serverbase import Server
from threading import Thread
from collections import defaultdict
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics



class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            ASR = self.test_backdoor_metrics()
            print(f"ASR : {ASR}")

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            


    def test_backdoor_metrics(self):
        total_ASR = 0.0
        total_auc = 0.0
        num_clients = self.num_clients  # 假设共10个客户端
        num_adv_clients = self.num_adv_clients  # 假设前5个客户端是恶意客户端

        for id in range(num_adv_clients, num_clients):
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

            if self.clients[id].global_protos is not None:
                with torch.no_grad():
                    for x, y in backdoorloaderfull:
                        if type(x) == type([]):
                            x[0] = x[0].to(self.clients[id].device)
                        else:
                            x = x.to(self.clients[id].device)
                        y = y.to(self.clients[id].device)
                        rep = self.clients[id].model.base(x)

                        output = float('inf') * torch.ones(y.shape[0], self.clients[id].num_classes).to(self.clients[id].device)
                        for i, r in enumerate(rep):
                            for j, pro in self.clients[id].global_protos.items():
                                if type(pro) != type([]):
                                    output[i, j] = self.clients[id].loss_mse(r, pro)

                        test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                        test_num += y.shape[0]
            else:
                test_acc = 0
                test_num =  1e-5
            
    #         with torch.no_grad():
    #             for i, (x, y) in enumerate(backdoorloaderfull):
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.clients[id].device)
    #                 else:
    #                     x = x.to(self.clients[id].device)
    #                 y = y.to(self.clients[id].device)
    #                 #print(y)
    #                 output = self.clients[id].model(x)
    #                 #print(output)
    # 
    #                 test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #                 # if(i == 0):
    #                 #     print(f"backdoor dataset label:{y}")
    #                 #     print(f"backdoor data predict:{torch.argmax(output, dim=1)}")
    #                 test_num += y.shape[0]
    # 
    #                 y_prob.append(output.detach().cpu().numpy())
    #                 nc = self.clients[id].num_classes
    #                 lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #                 y_true.append(lb)

            ASR = test_acc*1.0 / test_num
            # print(f"id {id} ASR: {ASR}")

            total_ASR += ASR  # 累加ASR

        avg_ASR = total_ASR / (num_clients - num_adv_clients)  # 计算平均ASR
            
        return avg_ASR

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


