from User.serveravg import FedAvg
from User.clientavg import clientAVG
import copy
import numpy as np
import torch

class FedBulyan(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.trim_ratio = 0.1

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        # 初始化全局模型
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        # 逐层处理神经网络参数
        for global_param, client_params in zip(
            self.global_model.parameters(),
            zip(*[model.parameters() for model in self.uploaded_models])
        ):
            # 将各客户端参数堆叠成张量 [num_clients, *layer_shape]
            stacked_params = torch.stack([p.data.clone() for p in client_params], dim=0)
            num_clients = stacked_params.shape[0]
            beta = int(self.trim_ratio * num_clients)  # 计算截断数量

            # 第一阶段：初步筛选（类似 Trimmed Mean）
            if beta > 0:
                # 沿客户端维度排序并截断
                sorted_params, _ = torch.sort(stacked_params, dim=0)
                trimmed_once = sorted_params[beta:-beta]
            else:
                trimmed_once = stacked_params

            # 第二阶段：基于距离的精确筛选
            # 计算初步筛选后的均值作为参考点
            if trimmed_once.shape[0] == 0:  # 处理全截断特殊情况
                trimmed_once = stacked_params
            trimmed_mean = trimmed_once.mean(dim=0)

            # 计算各客户端参数与参考点的L2距离
            flattened_params = stacked_params.view(num_clients, -1)  # 展平为向量
            flattened_ref = trimmed_mean.view(-1)
            distances = torch.norm(flattened_params - flattened_ref, p=2, dim=1)

            # 选择最接近的客户端（数量 = 原始数量 - 2*beta）
            k = num_clients - 2 * beta
            if k <= 0:
                k = 1  # 确保至少选择一个客户端
            selected_indices = torch.argsort(distances)[:k]
            selected_params = stacked_params[selected_indices]

            # 第三阶段：最终聚合（中位数）
            if selected_params.shape[0] > 0:
                median_values, _ = torch.median(selected_params, dim=0)
                global_param.data.copy_(median_values)
            else:  # 保底策略
                global_param.data.copy_(trimmed_mean)