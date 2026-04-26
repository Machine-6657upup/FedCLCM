import torch
import numpy as np
import time
from User.clientbase import Client
import torch.nn.functional as F


class clientCliDef(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=args.lr_head)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

        self.epsilon = args.adv_eps      # 0.1  # 扰动强度
        self.num_iter = args.adv_num_iter   # PGD迭代次数
        self.alpha = 2 * self.epsilon / self.num_iter  # 每次迭代的步长

        self.plocal_epochs = args.plocal_epochs
        self.num_adv_clients = args.num_adv_clients

        # 复用 FedCLCM 的可选客户端增强/对比学习参数
        self.lambda_cl = getattr(args, "lambda_cl", 0.0)
        self.aug_strength = getattr(args, "aug_strength", 0.0)

    def generate_augmented_views(self, x):
        """为对比学习生成增强视图（噪声增强）"""
        x_aug1 = x + torch.randn_like(x) * self.aug_strength
        x_aug1 = torch.clamp(x_aug1, -1, 1)
        x_aug2 = x + torch.randn_like(x) * self.aug_strength
        x_aug2 = torch.clamp(x_aug2, -1, 1)
        return x_aug1, x_aug2

    def contrastive_loss(self, h1, h2):
        """简化对比损失（MSE），与 FedCLCM 一致"""
        return F.mse_loss(h1, h2)

    def trigger_breaking_augment(self, x):
        """触发器破坏型增强（简化 Cutout / Random Erasing）"""
        if self.aug_strength <= 0:
            return x
        erase_prob = min(0.6, 2.0 * self.aug_strength)
        min_ratio = 0.05
        max_ratio = min(0.35, 0.05 + 1.5 * self.aug_strength)
        if max_ratio <= min_ratio:
            return x

        x = x.clone()
        b, c, h, w = x.shape
        for i in range(b):
            if torch.rand(1).item() > erase_prob:
                continue
            erase_ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
            cut_h = max(1, int(h * erase_ratio))
            cut_w = max(1, int(w * erase_ratio))
            y0 = torch.randint(0, max(1, h - cut_h + 1), (1,)).item()
            x0 = torch.randint(0, max(1, w - cut_w + 1), (1,)).item()
            x[i, :, y0:y0 + cut_h, x0:x0 + cut_w] = torch.empty(
                (c, cut_h, cut_w), device=x.device
            ).uniform_(-1.0, 1.0)
        return x

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        is_benign = self.id >= self.num_adv_clients

        # self adv clean
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            x_base = self.trigger_breaking_augment(x) if is_benign else x
            x_orig = x_base.clone().detach()
            delta = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
            adv_x = x_orig + delta
            adv_x = torch.clamp(adv_x, -1, 1)

            self.model.eval()

            for _ in range(self.num_iter):
                adv_x.requires_grad_(True)
                
                # 前向计算
                with torch.enable_grad():
                    outputs = self.model(adv_x)
                    loss = F.cross_entropy(outputs, y)
                
                # 计算输入梯度
                grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
                
                # 更新对抗扰动
                adv_x = adv_x.detach() + self.alpha * grad.sign()
                delta = adv_x - x_orig
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)  # 投影到epsilon邻域
                # adv_x = x_orig + delta
                adv_x = torch.clamp(x_orig + delta, -1, 1).detach()  # 这里改到了-1到1

            self.model.train()
            for param in self.model.base.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True

            output = self.model(adv_x)
            loss = self.loss(output, y)
            self.optimizer_per.zero_grad()
            loss.backward()
            self.optimizer_per.step()


        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()          

        max_local_epochs = self.local_epochs

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x_main = self.trigger_breaking_augment(x) if is_benign else x
                output = self.model(x_main)
                ce_loss = self.loss(output, y)

                if is_benign and self.lambda_cl > 0:
                    x_aug1, x_aug2 = self.generate_augmented_views(x_main)
                    h1 = self.model.base(x_aug1)
                    h2 = self.model.base(x_aug2)
                    cl_loss = self.contrastive_loss(h1, h2)
                    loss = ce_loss + self.lambda_cl * cl_loss
                else:
                    loss = ce_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()