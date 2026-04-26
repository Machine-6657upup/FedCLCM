# servertrimmed.py
from User.serveravg import FedAvg
from User.clientavg import clientAVG
import copy
import numpy as np
import torch

class FedTrimmed(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.trim_ratio = 0.1  # 修剪比例

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
            k = int(self.trim_ratio * num_clients)
            
            # 沿客户端维度排序
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            # 执行截断（去除前k个和后k个）
            trimmed_params = sorted_params[k:-k] if k > 0 else sorted_params
            # 计算均值并更新全局参数
            global_param.data = trimmed_params.mean(dim=0)
