from User.serveravg import FedAvg
from User.clientavg import clientAVG
import copy
import numpy as np
import torch

class FedMedian(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

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
            
            # 沿客户端维度计算中位数
            median_values, _ = torch.median(stacked_params, dim=0)
            
            # 更新全局参数
            global_param.data = median_values