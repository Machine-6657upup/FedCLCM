import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import torch.nn.functional as F
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.preprocessing import label_binarize

random.seed(42)


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate

        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = 0
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = 0

        self.num_adv_clients = args.num_adv_clients

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
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
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc


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
            # print(f"id {id} ASR: {ASR}")

            total_ASR += ASR  # 累加ASR
            total_auc += auc

        avg_ASR = total_ASR / (num_clients - num_adv_clients)  # 计算平均ASR
        avg_auc = total_auc / (num_clients - num_adv_clients)
            
        return avg_ASR, avg_auc

    def _parse_thresholds(self, s, defaults):
        if s is None:
            return defaults
        s = str(s).strip()
        if not s:
            return defaults
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return out if out else defaults

    def _collect_logits_stats(self, loader, model, device):
        max_probs = []
        entropies = []
        margins = []
        correct = []
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                top2 = torch.topk(probs, k=2, dim=1).values
                max_prob = top2[:, 0]
                margin = top2[:, 0] - top2[:, 1]
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

                pred = torch.argmax(probs, dim=1)
                correct.append((pred == y).detach().cpu())
                max_probs.append(max_prob.detach().cpu())
                entropies.append(entropy.detach().cpu())
                margins.append(margin.detach().cpu())
                total += y.shape[0]

        max_probs = torch.cat(max_probs).numpy() if max_probs else np.array([])
        entropies = torch.cat(entropies).numpy() if entropies else np.array([])
        margins = torch.cat(margins).numpy() if margins else np.array([])
        correct = torch.cat(correct).numpy() if correct else np.array([])
        return max_probs, entropies, margins, correct, total

    def test_backdoor_logit_stats(self):
        """
        验证“后门输入 logits 更平滑/更不自信”的假设：
        - 统计 clean vs backdoor 的 max softmax / entropy / margin 分布
        - 计算可分性（AUC）
        - 基于阈值的拒绝机制评估
        """
        num_clients = self.num_clients
        num_adv_clients = self.num_adv_clients

        conf_th = self._parse_thresholds(
            getattr(self.args, "logit_reject_thresholds", None),
            defaults=[0.5, 0.6, 0.7, 0.8]
        )
        ent_th = self._parse_thresholds(
            getattr(self.args, "logit_entropy_thresholds", None),
            defaults=[1.0, 1.5, 2.0]
        )
        margin_th = self._parse_thresholds(
            getattr(self.args, "logit_margin_thresholds", None),
            defaults=[0.1, 0.2, 0.3]
        )

        clean_max, clean_ent, clean_margin, clean_correct, clean_total = [], [], [], [], 0
        bd_max, bd_ent, bd_margin, bd_correct, bd_total = [], [], [], [], 0

        for id in range(num_adv_clients, num_clients):
            client = self.clients[id]
            clean_loader = client.load_test_data()

            backdoor_data_dir = os.path.join('./dataset', client.dataset, 'test/')
            backdoor_file = os.path.join(backdoor_data_dir, f'{id}_backdoored.npz')
            with open(backdoor_file, 'rb') as f:
                backdoor_data = np.load(f, allow_pickle=True)['data'].tolist()
            X_backdoor = torch.Tensor(backdoor_data['x']).type(torch.float32)
            y_backdoor = torch.Tensor(backdoor_data['y']).type(torch.int64)
            backdoor_data = [(x, y) for x, y in zip(X_backdoor, y_backdoor)]
            backdoor_loader = DataLoader(backdoor_data, client.batch_size, drop_last=False, shuffle=True)

            cm, ce, cmar, ccor, ctot = self._collect_logits_stats(
                clean_loader, client.model, client.device
            )
            bm, be, bmar, bcor, btot = self._collect_logits_stats(
                backdoor_loader, client.model, client.device
            )

            if cm.size:
                clean_max.append(cm); clean_ent.append(ce); clean_margin.append(cmar); clean_correct.append(ccor)
                clean_total += ctot
            if bm.size:
                bd_max.append(bm); bd_ent.append(be); bd_margin.append(bmar); bd_correct.append(bcor)
                bd_total += btot

        if not clean_max or not bd_max:
            print("Logit analysis skipped: empty clean/backdoor stats.")
            return

        clean_max = np.concatenate(clean_max)
        clean_ent = np.concatenate(clean_ent)
        clean_margin = np.concatenate(clean_margin)
        clean_correct = np.concatenate(clean_correct)
        bd_max = np.concatenate(bd_max)
        bd_ent = np.concatenate(bd_ent)
        bd_margin = np.concatenate(bd_margin)
        bd_correct = np.concatenate(bd_correct)

        def _stat(x):
            return float(x.mean()), float(x.std()), float(np.quantile(x, 0.1)), float(np.quantile(x, 0.5)), float(np.quantile(x, 0.9))

        print("\n[Logit Analysis] Clean vs Backdoor distributions")
        c_mean, c_std, c_p10, c_p50, c_p90 = _stat(clean_max)
        b_mean, b_std, b_p10, b_p50, b_p90 = _stat(bd_max)
        print(f"MaxProb clean: mean={c_mean:.4f} std={c_std:.4f} p10={c_p10:.4f} p50={c_p50:.4f} p90={c_p90:.4f}")
        print(f"MaxProb backdoor: mean={b_mean:.4f} std={b_std:.4f} p10={b_p10:.4f} p50={b_p50:.4f} p90={b_p90:.4f}")

        c_mean, c_std, c_p10, c_p50, c_p90 = _stat(clean_ent)
        b_mean, b_std, b_p10, b_p50, b_p90 = _stat(bd_ent)
        print(f"Entropy clean: mean={c_mean:.4f} std={c_std:.4f} p10={c_p10:.4f} p50={c_p50:.4f} p90={c_p90:.4f}")
        print(f"Entropy backdoor: mean={b_mean:.4f} std={b_std:.4f} p10={b_p10:.4f} p50={b_p50:.4f} p90={b_p90:.4f}")

        c_mean, c_std, c_p10, c_p50, c_p90 = _stat(clean_margin)
        b_mean, b_std, b_p10, b_p50, b_p90 = _stat(bd_margin)
        print(f"Margin clean: mean={c_mean:.4f} std={c_std:.4f} p10={c_p10:.4f} p50={c_p50:.4f} p90={c_p90:.4f}")
        print(f"Margin backdoor: mean={b_mean:.4f} std={b_std:.4f} p10={b_p10:.4f} p50={b_p50:.4f} p90={b_p90:.4f}")

        y = np.concatenate([np.zeros_like(clean_max), np.ones_like(bd_max)])
        score_conf = np.concatenate([1 - clean_max, 1 - bd_max])
        score_ent = np.concatenate([clean_ent, bd_ent])
        score_margin = np.concatenate([1 - clean_margin, 1 - bd_margin])
        try:
            auc_conf = metrics.roc_auc_score(y, score_conf)
            auc_ent = metrics.roc_auc_score(y, score_ent)
            auc_margin = metrics.roc_auc_score(y, score_margin)
            print(f"AUC (1-maxprob): {auc_conf:.4f}, AUC (entropy): {auc_ent:.4f}, AUC (1-margin): {auc_margin:.4f}")
        except Exception as e:
            print(f"AUC compute failed: {e}")

        print("\n[Logit Analysis] Reject by maxprob")
        for th in conf_th:
            clean_accept = clean_max >= th
            bd_accept = bd_max >= th
            clean_acc = clean_correct[clean_accept].mean() if clean_accept.any() else 0.0
            bd_asr = bd_correct[bd_accept].mean() if bd_accept.any() else 0.0
            print(f"th={th:.2f} clean_accept={clean_accept.mean():.3f} clean_acc={clean_acc:.3f} bd_accept={bd_accept.mean():.3f} bd_asr={bd_asr:.3f}")

        print("\n[Logit Analysis] Reject by entropy")
        for th in ent_th:
            clean_accept = clean_ent <= th
            bd_accept = bd_ent <= th
            clean_acc = clean_correct[clean_accept].mean() if clean_accept.any() else 0.0
            bd_asr = bd_correct[bd_accept].mean() if bd_accept.any() else 0.0
            print(f"th={th:.2f} clean_accept={clean_accept.mean():.3f} clean_acc={clean_acc:.3f} bd_accept={bd_accept.mean():.3f} bd_asr={bd_asr:.3f}")

        print("\n[Logit Analysis] Reject by margin")
        for th in margin_th:
            clean_accept = clean_margin >= th
            bd_accept = bd_margin >= th
            clean_acc = clean_correct[clean_accept].mean() if clean_accept.any() else 0.0
            bd_asr = bd_correct[bd_accept].mean() if bd_accept.any() else 0.0
            print(f"th={th:.2f} clean_accept={clean_accept.mean():.3f} clean_acc={clean_acc:.3f} bd_accept={bd_accept.mean():.3f} bd_asr={bd_asr:.3f}")