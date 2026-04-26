# servermk.py修正版
from User.serveravg import FedAvg
from User.clientavg import clientAVG
import copy
import numpy as np
import torch

class FedMK(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientAVG)
        self.m = 8  # 选择的客户端数量
        self.k = 7  # 每个客户端考虑的最近邻居数
    
    def _get_flatten_params(self, model):
        """将模型参数展平为一维numpy数组"""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        n = len(self.uploaded_models)
        distances = np.zeros((n, n))
        
        # 预计算所有展平参数
        flat_params = [self._get_flatten_params(model) for model in self.uploaded_models]
        
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.linalg.norm(flat_params[i] - flat_params[j])
        
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])[:self.k+1]  # 包含自己的k+1个最近距离
            scores.append(np.sum(sorted_dists))
        
        # 选择分数最低的m个客户端
        selected_indices = np.argsort(scores)[:self.m]

        # 初始化全局模型参数为零
        self.global_model.load_state_dict(self.uploaded_models[0].state_dict())
        for param in self.global_model.parameters():
            param.data.zero_()
        
        # 聚合参数（按名称对应，处理设备）
        for idx in selected_indices:
            local_state = self.uploaded_models[idx].state_dict()
            for name, param in self.global_model.named_parameters():
                if name in local_state:
                    # 确保参数在相同设备上
                    local_param = local_state[name].to(self.device)
                    param.data.add_(local_param / len(selected_indices))