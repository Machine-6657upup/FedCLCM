import torch
import time
from User.clientapple import clientAPPLE
from User.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data


class APPLE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_clients(clientAPPLE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.client_models = [c.model_c for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.clients:
                client.train(i)

            ASR, attack_auc = self.test_backdoor_metrics()
            print(f"ASR : {ASR}")
            print(f"attack_auc : {attack_auc}")


            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

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
            self.set_new_clients(clientAPPLE)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        self.args.num_clients = self.num_clients
        

    def send_models(self):
        assert (len(self.clients) > 0)

        self.client_models = [c.model_c for c in self.clients]
        for client in self.clients:
            start_time = time.time()
            
            client.set_models(self.client_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

