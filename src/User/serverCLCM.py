import random
import time
from User.clientCLCM import clientCLCM
from User.serverbase import Server
from threading import Thread
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import copy


class FedCLCM(Server):
    """
    FedCLCM Server: 使用可学习通道Mask的鲁棒聚合服务器
    - 基于更新的通道级方差检测可疑通道
    - 对可疑通道进行降权处理
    - 结合Trimmed Mean进行鲁棒聚合
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.set_clients(clientCLCM)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []
        self.beta = args.rt_beta  # Trimmed Mean的截断比例
        
        # 可学习通道Mask参数
        self.mask_tau = args.mask_tau  # 方差阈值倍数
        self.mask_alpha = args.mask_alpha  # 可疑通道的降权系数
        self.channel_mask = {}  # 存储每层的通道mask
        self.enable_channel_mask = args.enable_channel_mask  # 是否启用通道mask

        # ===== 一致性加权（cosine gating）=====
        # 用于降低方向异常的更新权重（避免硬裁剪导致误伤）
        self.cosine_gate = getattr(args, "cosine_gate", False)
        self.cosine_gate_threshold = getattr(args, "cosine_gate_threshold", 0.3)
        self.cosine_gate_alpha = getattr(args, "cosine_gate_alpha", 0.5)

        # ===== 分层 Trim（高层更强 / 低层更弱）=====
        self.trim_high_layers = getattr(args, "trim_high_layers", "")
        self.trim_beta_high = getattr(args, "trim_beta_high", None)
        self.trim_beta_low = getattr(args, "trim_beta_low", None)
        if isinstance(self.trim_high_layers, str):
            self.trim_high_layers = [s.strip() for s in self.trim_high_layers.split(",") if s.strip()]
        
        # 存储上一轮的全局模型（用于计算更新）
        self.prev_global_model = None

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            
            # 客户端训练
            for client in self.selected_clients:
                client.train()
            
            # 测试后门攻击效果
            ASR, attack_auc = self.test_backdoor_metrics()
            print(f"ASR: {ASR}")
            print(f"attack_auc: {attack_auc}")

            if getattr(self.args, "logit_analysis", False):
                self.test_backdoor_logit_stats()
            
            # 接收客户端模型
            self.receive_models()
            
            # 如果启用通道mask，先更新mask
            if self.enable_channel_mask and i > 0:
                self.update_channel_mask()
            
            # 聚合参数
            self.aggregate_parameters()
            
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        
        self.save_results()
        
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCLCM)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def get_current_global_base(self):
        """
        获取当前全局base模型
        """
        if hasattr(self.global_model, 'base'):
            return self.global_model.base
        else:
            return self.global_model

    def compute_updates(self):
        """
        计算每个客户端的更新（相对于上一轮的全局模型）
        返回: updates列表，每个元素是一个模型的更新
        """
        if self.prev_global_model is None:
            return None
        
        updates = []
        prev_global_base = self.prev_global_model
        
        for client_model in self.uploaded_models:
            update = {}
            for (name, prev_param), (_, client_param) in zip(
                prev_global_base.named_parameters(),
                client_model.named_parameters()
            ):
                update[name] = client_param.data - prev_param.data
            updates.append(update)
        
        return updates

    def compute_client_weights(self):
        """
        一致性加权（cosine gating）
        - 计算每个客户端更新与均值更新的余弦相似度
        - 相似度过低则降低权重（不直接丢弃）
        """
        if (not self.cosine_gate) or self.prev_global_model is None:
            return None

        updates = self.compute_updates()
        if updates is None:
            return None

        # 将每个客户端更新展平成向量
        vecs = []
        for update in updates:
            flat = []
            for _, t in update.items():
                flat.append(t.detach().reshape(-1))
            vecs.append(torch.cat(flat))

        if not vecs:
            return None

        mat = torch.stack(vecs, dim=0)
        mean_vec = mat.mean(dim=0)
        eps = 1e-12
        mean_norm = torch.norm(mean_vec) + eps

        # 计算余弦相似度
        mat_norm = torch.norm(mat, dim=1) + eps
        cos = (mat @ mean_vec) / (mat_norm * mean_norm)

        # gating：低相似度降权
        weights = torch.ones_like(cos, device=cos.device)
        low_mask = cos < self.cosine_gate_threshold
        weights[low_mask] = self.cosine_gate_alpha
        return weights

    def _get_layer_beta(self, name):
        """
        分层 trim：高层使用更强的 beta，高层由 trim_high_layers 指定
        """
        if self.trim_beta_high is None and self.trim_beta_low is None:
            return self.beta

        is_high = any(name.startswith(prefix) for prefix in self.trim_high_layers)
        if is_high:
            return self.trim_beta_high if self.trim_beta_high is not None else self.beta
        return self.trim_beta_low if self.trim_beta_low is not None else self.beta

    def update_channel_mask(self):
        """
        基于更新的通道级方差更新可学习通道mask
        方差异常高的通道被视为可疑（可能被后门攻击影响）
        """
        updates = self.compute_updates()
        if updates is None:
            return
        
        prev_global_base = self.prev_global_model
        
        # 遍历每一层
        for name, param in prev_global_base.named_parameters():
            # 只对卷积层和全连接层的权重进行通道mask
            if 'weight' not in name or len(param.shape) < 2:
                continue
            
            # 获取通道数（通常是第0维）
            num_channels = param.shape[0]
            
            # 收集所有客户端在该层的更新
            layer_updates = [update[name] for update in updates]
            
            # 计算每个通道的方差
            variances = []
            for c in range(num_channels):
                # 提取第c个通道的所有参数
                channel_updates = torch.stack([
                    upd[c].flatten() for upd in layer_updates
                ])  # shape: [num_clients, channel_params]
                
                # 计算该通道跨客户端的方差
                var = channel_updates.var(dim=0).mean().item()
                variances.append(var)
            
            # 构造mask
            variances = np.array(variances)
            threshold = np.median(variances) * self.mask_tau
            
            mask = torch.ones(num_channels, device=self.device)
            for c, v in enumerate(variances):
                if v > threshold:
                    mask[c] = self.mask_alpha  # 降权可疑通道
            
            self.channel_mask[name] = mask
            
            # 打印统计信息
            num_suspicious = (mask < 1.0).sum().item()
            if num_suspicious > 0:
                print(f"Layer {name}: {num_suspicious}/{num_channels} channels masked")

    def apply_channel_mask(self):
        """
        将通道mask应用到全局模型
        """
        if not self.enable_channel_mask or len(self.channel_mask) == 0:
            return
        
        current_global_base = self.get_current_global_base()
        
        for name, param in current_global_base.named_parameters():
            if name in self.channel_mask:
                mask = self.channel_mask[name]
                # 将mask扩展到参数的完整形状
                mask_shape = [1] * len(param.shape)
                mask_shape[0] = -1  # 通道维度
                mask = mask.view(*mask_shape)
                param.data *= mask

    def aggregate_parameters(self):
        """
        使用Trimmed Mean聚合客户端的base模型参数
        在聚合时应用通道mask（修复版本：避免破坏参数）
        """
        assert (len(self.uploaded_models) > 0)

        # 计算一致性权重（需要上一轮的 prev_global_model）
        client_weights = self.compute_client_weights()

        # 保存当前全局模型作为下一轮的prev_global_model
        current_global_base = self.get_current_global_base()
        self.prev_global_model = copy.deepcopy(current_global_base)
        
        # Trimmed Mean聚合（在聚合时应用mask）
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for (name, global_param), client_params in zip(
            self.global_model.named_parameters(),
            zip(*[model.parameters() for model in self.uploaded_models])
        ):
            # 将各客户端该层参数堆叠成张量 [num_clients, layer_shape...]
            stacked_params = torch.stack([p.data.clone() for p in client_params], dim=0)

            # 一致性加权（cosine gating）
            if client_weights is not None:
                w_shape = [client_weights.shape[0]] + [1] * (stacked_params.dim() - 1)
                stacked_params = stacked_params * client_weights.view(*w_shape)
            
            # 在聚合前应用通道mask（如果启用且该层有mask）
            if self.enable_channel_mask and name in self.channel_mask:
                mask = self.channel_mask[name]  # shape: [num_channels]
                # 将mask扩展到可以broadcast到stacked_params的形状
                # stacked_params: [num_clients, num_channels, ...]
                # mask需要扩展为: [1, num_channels, 1, 1, ...] (与stacked_params的后N-1维匹配)
                mask_shape = [1, -1] + [1] * (len(stacked_params.shape) - 2)
                mask_expanded = mask.view(mask_shape)
                # 对每个客户端的可疑通道参数降权（直接在聚合前应用）
                stacked_params = stacked_params * mask_expanded
            
            num_clients = stacked_params.shape[0]
            beta = self._get_layer_beta(name)
            k = int(beta * num_clients)
            
            # 沿客户端维度排序
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            # 执行截断（去除前k个和后k个）
            trimmed_params = sorted_params[k:-k] if k > 0 else sorted_params
            # 计算均值并更新全局参数
            global_param.data = trimmed_params.mean(dim=0)

    def receive_models(self):
        """
        接收客户端上传的base模型
        """
        assert (len(self.selected_clients) > 0)
        
        active_clients = random.sample(
            self.selected_clients, 
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )
        
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
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
