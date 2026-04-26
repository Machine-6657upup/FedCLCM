# clientflip.py
import copy
import torch
import numpy as np
import time
from User.clientbase import Client

class clientFLIP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.trigger_size = 5  # 需要添加到参数中
        self.trigger = None
        self.distance_matrix = {}
        self.promising_pairs = []

    def generate_trigger(self, img_shape):
        """生成与输入同尺寸的触发模式"""
        # 假设img_shape为 (C, H, W)
        mask = torch.zeros(img_shape)
        pattern = torch.zeros(img_shape)
        
        # 在右下角创建触发模式
        s = self.trigger_size
        mask[:, -s:, -s:] = 1.0  # 触发区域掩码
        pattern[:, -s:, -s:] = 1.0  # 白色方块触发模式
        return mask, pattern

    def apply_trigger(self, x, mask, pattern):
        """应用触发模式（支持批量处理）"""
        # x的形状： (batch, C, H, W)
        # mask/pattern形状： (C, H, W) 或 (1, H, W)
        return (1 - mask) * x + mask * pattern

    def adversarial_train(self, trainloader):
        """生成对抗样本"""
        # 获取样本尺寸
        sample, _ = next(iter(trainloader))
        img_shape = sample.shape[1:]  # (C, H, W)
        
        # 生成触发模式
        mask, pattern = self.generate_trigger(img_shape)
        mask = mask.to(self.device)
        pattern = pattern.to(self.device)
        
        # 生成对抗样本
        adv_examples, adv_labels = [], []
        for data, target in trainloader:
            data = data.to(self.device)
            # 应用触发模式
            adv_data = self.apply_trigger(data, mask, pattern)
            adv_examples.append(adv_data.cpu())
            adv_labels.append(target)
        
        return torch.cat(adv_examples), torch.cat(adv_labels)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        # 生成对抗样本
        adv_data, adv_labels = self.adversarial_train(trainloader)
        adv_dataset = torch.utils.data.TensorDataset(adv_data, adv_labels)
        
        # 合并数据集
        combined_dataset = torch.utils.data.ConcatDataset([
            trainloader.dataset,
            adv_dataset
        ])
        
        combined_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )


        start_time = time.time()
        
        for epoch in range(self.local_epochs):
            for x, y in combined_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time