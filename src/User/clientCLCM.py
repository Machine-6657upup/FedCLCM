import torch
import numpy as np
import time
import copy
from User.clientbase import Client
import torch.nn.functional as F


class clientCLCM(Client):
    """
    FedCLCM Client: 结合对比学习的个性化联邦学习客户端
    - Benign客户端: 使用对比学习训练base，交替训练base和head
    - Malicious客户端: 使用后门数据，交替训练base和head
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 为base和head分别创建优化器
        self.optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_base = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_base, 
            gamma=args.learning_rate_decay_gamma
        )
        
        self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=args.lr_head)
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_head, 
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_epochs = args.plocal_epochs
        self.num_adv_clients = args.num_adv_clients
        
        # 对比学习参数
        self.lambda_cl = args.lambda_cl  # 对比学习损失权重
        # 数据增强强度：同时用于 CL 噪声与触发器破坏型增强（cutout）
        self.aug_strength = args.aug_strength

        # ===== PGD 对抗训练参数（复用已有 FedSD/FedRT 参数，不改 main.py）=====
        # 关键策略：只对良性客户端启用（防止给攻击者“强化训练”）
        self.adv_eps = getattr(args, "adv_eps", 0.0)
        self.adv_num_iter = int(getattr(args, "adv_num_iter", 0))
        self.adv_step = (2 * self.adv_eps / self.adv_num_iter) if (self.adv_num_iter and self.adv_eps > 0) else 0.0

        # BDPFL/PFL-ALP inspired client-side purification. Disabled by default.
        self.purify_enable = bool(getattr(args, "purify_enable", False))
        self.purify_beta = float(getattr(args, "purify_beta", 0.0))
        self.purify_feature_beta = float(getattr(args, "purify_feature_beta", 0.0))
        self.purify_logit_beta = float(getattr(args, "purify_logit_beta", 0.0))
        self.purify_temperature = float(getattr(args, "purify_temperature", 2.0))
        self.purify_start_round = int(getattr(args, "purify_start_round", 1))
        self.purify_teacher_momentum = float(getattr(args, "purify_teacher_momentum", 0.9))
        self.purify_teacher_cpu_half = bool(getattr(args, "purify_teacher_cpu_half", True))
        # Keep the default conservative. Higher layers carry more backdoor
        # semantics, and old CLCM logs show over-constraining the base can hurt.
        layers = getattr(args, "purify_layers", "layer4")
        self.purify_layers = [s.strip() for s in str(layers).split(",") if s.strip()]
        self.purify_teacher_state = None

    def set_dynamic_hparams(self, params):
        if "lambda_cl" in params:
            self.lambda_cl = float(params["lambda_cl"])
        if "aug_strength" in params:
            self.aug_strength = float(params["aug_strength"])
        if "adv_eps" in params:
            self.adv_eps = float(params["adv_eps"])
        if "adv_num_iter" in params:
            self.adv_num_iter = int(float(params["adv_num_iter"]))
        if "purify_beta" in params:
            self.purify_beta = float(params["purify_beta"])
        if "purify_feature_beta" in params:
            self.purify_feature_beta = float(params["purify_feature_beta"])
        if "purify_logit_beta" in params:
            self.purify_logit_beta = float(params["purify_logit_beta"])
        if "purify_start_round" in params:
            self.purify_start_round = int(float(params["purify_start_round"]))
        if "purify_teacher_momentum" in params:
            self.purify_teacher_momentum = float(params["purify_teacher_momentum"])
        if "lr" in params:
            for group in self.optimizer_base.param_groups:
                group["lr"] = float(params["lr"])
        if "lr_head" in params:
            for group in self.optimizer_head.param_groups:
                group["lr"] = float(params["lr_head"])
        self.adv_step = (2 * self.adv_eps / self.adv_num_iter) if (self.adv_num_iter and self.adv_eps > 0) else 0.0

    def _pack_teacher_state(self, model):
        state = {}
        for name, value in model.state_dict().items():
            tensor = value.detach().cpu().clone()
            if self.purify_teacher_cpu_half and torch.is_floating_point(tensor):
                tensor = tensor.half()
            state[name] = tensor
        return state

    def _update_purify_teacher(self):
        if not self.purify_enable:
            return
        new_state = self._pack_teacher_state(self.model)
        if self.purify_teacher_state is None or self.purify_teacher_momentum <= 0:
            self.purify_teacher_state = new_state
            return

        momentum = self.purify_teacher_momentum
        mixed_state = {}
        for name, new_tensor in new_state.items():
            old_tensor = self.purify_teacher_state.get(name)
            if old_tensor is None or old_tensor.shape != new_tensor.shape:
                mixed_state[name] = new_tensor
                continue
            if torch.is_floating_point(new_tensor):
                mixed = old_tensor.float().mul(momentum).add(
                    new_tensor.float(), alpha=(1.0 - momentum)
                )
                if self.purify_teacher_cpu_half:
                    mixed = mixed.half()
                mixed_state[name] = mixed
            else:
                mixed_state[name] = new_tensor
        self.purify_teacher_state = mixed_state

    def _build_purify_teacher(self, is_benign):
        if not (
            self.purify_enable
            and is_benign
            and self.purify_teacher_state is not None
            and self.train_time_cost["num_rounds"] >= self.purify_start_round
            and (self.purify_beta > 0 or self.purify_feature_beta > 0 or self.purify_logit_beta > 0)
        ):
            return None
        teacher = copy.deepcopy(self.model)
        teacher.load_state_dict(self.purify_teacher_state, strict=True)
        teacher.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def _forward_base_with_activations(self, base, x, eval_mode=False):
        modules = dict(base.named_modules())
        activations = {}
        handles = []
        was_training = base.training

        def make_hook(name):
            def hook(_module, _inputs, output):
                activations[name] = output
            return hook

        for name in self.purify_layers:
            module = modules.get(name)
            if module is not None:
                handles.append(module.register_forward_hook(make_hook(name)))

        try:
            if eval_mode:
                base.eval()
            features = base(x)
        finally:
            for handle in handles:
                handle.remove()
            if eval_mode and was_training:
                base.train()
        return features, activations

    def _attention_map(self, feat):
        if feat.dim() >= 4:
            att = feat.pow(2).mean(dim=1)
        else:
            att = feat
        att = att.flatten(start_dim=1)
        return F.normalize(att, p=2, dim=1, eps=1e-6)

    def purification_loss(self, x, teacher):
        # Use eval-mode BN for both networks in the purification forward.
        # Otherwise identical weights still produce a large loss because the
        # student uses batch statistics while the teacher uses running stats.
        cur_feat, cur_acts = self._forward_base_with_activations(self.model.base, x, eval_mode=True)
        with torch.no_grad():
            tea_feat, tea_acts = self._forward_base_with_activations(teacher.base, x, eval_mode=True)

        loss = x.new_tensor(0.0)
        used_terms = 0

        if self.purify_beta > 0:
            for name in self.purify_layers:
                if name not in cur_acts or name not in tea_acts:
                    continue
                cur_att = self._attention_map(cur_acts[name])
                tea_att = self._attention_map(tea_acts[name])
                loss = loss + self.purify_beta * F.mse_loss(cur_att, tea_att)
                used_terms += 1

        if self.purify_feature_beta > 0:
            cur_norm = F.normalize(cur_feat.flatten(start_dim=1), p=2, dim=1, eps=1e-6)
            tea_norm = F.normalize(tea_feat.flatten(start_dim=1), p=2, dim=1, eps=1e-6)
            loss = loss + self.purify_feature_beta * F.mse_loss(cur_norm, tea_norm)
            used_terms += 1

        if self.purify_logit_beta > 0:
            temp = max(self.purify_temperature, 1e-6)
            cur_logits = self.model.head(cur_feat)
            with torch.no_grad():
                tea_logits = teacher.head(tea_feat)
            kl = F.kl_div(
                F.log_softmax(cur_logits / temp, dim=1),
                F.softmax(tea_logits / temp, dim=1),
                reduction="batchmean",
            ) * (temp * temp)
            loss = loss + self.purify_logit_beta * kl
            used_terms += 1

        if used_terms == 0:
            cur_norm = F.normalize(cur_feat.flatten(start_dim=1), p=2, dim=1, eps=1e-6)
            tea_norm = F.normalize(tea_feat.flatten(start_dim=1), p=2, dim=1, eps=1e-6)
            loss = loss + self.purify_beta * F.mse_loss(cur_norm, tea_norm)

        return loss

    def generate_augmented_views(self, x):
        """
        为对比学习生成增强视图
        使用随机噪声增强
        """
        # 第一个增强视图
        x_aug1 = x + torch.randn_like(x) * self.aug_strength
        x_aug1 = torch.clamp(x_aug1, -1, 1)
        
        # 第二个增强视图
        x_aug2 = x + torch.randn_like(x) * self.aug_strength
        x_aug2 = torch.clamp(x_aug2, -1, 1)
        
        return x_aug1, x_aug2

    def contrastive_loss(self, h1, h2):
        """
        计算对比损失（简化版，使用MSE）
        鼓励同一样本的不同增强视图在特征空间中保持一致
        """
        return F.mse_loss(h1, h2)

    def trigger_breaking_augment(self, x):
        """
        触发器破坏型增强（简化 Cutout / Random Erasing）。
        目的：在不改变 main.py 的前提下，增加对触发器的鲁棒性。
        仅在良性客户端启用。
        """
        if self.aug_strength <= 0:
            return x

        # 概率和擦除面积随 aug_strength 缩放
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
            # 用随机噪声填充（输入已归一化到 [-1, 1]）
            x[i, :, y0:y0 + cut_h, x0:x0 + cut_w] = torch.empty(
                (c, cut_h, cut_w), device=x.device
            ).uniform_(-1.0, 1.0)
        return x

    def pgd_attack(self, x, y):
        """
        PGD 对抗样本生成（输入范围假设已归一化到 [-1, 1]）。
        - 仅用于良性客户端的鲁棒训练
        - 不更新模型参数，只生成对抗输入
        """
        if self.adv_eps <= 0 or self.adv_num_iter <= 0:
            return x

        x_orig = x.detach()
        # 随机初始化
        delta = torch.zeros_like(x_orig).uniform_(-self.adv_eps, self.adv_eps)
        adv_x = torch.clamp(x_orig + delta, -1, 1)

        # 生成对抗样本时用 eval 更稳定（与 clientCliDef 一致的做法）
        self.model.eval()
        for _ in range(self.adv_num_iter):
            adv_x.requires_grad_(True)
            with torch.enable_grad():
                outputs = self.model(adv_x)
                loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
            adv_x = adv_x.detach() + (self.adv_step * grad.sign())
            delta = adv_x - x_orig
            delta = torch.clamp(delta, -self.adv_eps, self.adv_eps)
            adv_x = torch.clamp(x_orig + delta, -1, 1).detach()

        self.model.train()
        return adv_x

    def train(self):
        """
        客户端训练主函数
        - Benign客户端（id >= num_adv_clients）: 使用对比学习训练
        - Malicious客户端（id < num_adv_clients）: 正常训练（可能使用后门数据）
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        
        self.model.train()
        
        is_benign = self.id >= self.num_adv_clients
        teacher_model = self._build_purify_teacher(is_benign)
        # ===== 只在 Head 阶段启用 PGD（良性端）=====
        # 目的：在保证一定防御效果的同时，尽量减少鲁棒训练对 Base 学习与 ACC 的压制
        enable_pgd_head = is_benign and (self.adv_eps > 0) and (self.adv_num_iter > 0)
        
        # ========== 阶段1: 训练Head（冻结Base）==========
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

                # === 关键：仅良性客户端在 head 阶段做鲁棒训练（PGD）===
                x_base = self.trigger_breaking_augment(x) if is_benign else x
                x_in = self.pgd_attack(x_base, y) if enable_pgd_head else x_base

                output = self.model(x_in)
                loss = self.loss(output, y)
                
                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_head.step()
        
        # ========== 阶段2: 训练Base（冻结Head）==========
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False
        
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Base 阶段不使用 PGD（为提升 ACC，避免鲁棒训练压制 base 表示学习）
                x_main = self.trigger_breaking_augment(x) if is_benign else x
                
                if is_benign:
                    # Benign客户端：使用对比学习
                    # 生成两个增强视图
                    x_aug1, x_aug2 = self.generate_augmented_views(x_main)
                    
                    # 提取base特征
                    h1 = self.model.base(x_aug1)
                    h2 = self.model.base(x_aug2)
                    
                    # 计算对比损失
                    cl_loss = self.contrastive_loss(h1, h2)
                    
                    # 计算分类损失
                    output = self.model(x_main)
                    ce_loss = self.loss(output, y)
                    
                    # 总损失 = 分类损失 + 对比损失
                    loss = ce_loss + self.lambda_cl * cl_loss
                    if teacher_model is not None:
                        loss = loss + self.purification_loss(x, teacher_model)
                else:
                    # Malicious客户端：正常训练（可能包含后门数据）
                    output = self.model(x)
                    loss = self.loss(output, y)
                
                self.optimizer_base.zero_grad()
                loss.backward()
                self.optimizer_base.step()
        
        # 恢复所有参数的梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler_base.step()
            self.learning_rate_scheduler_head.step()

        if is_benign:
            self._update_purify_teacher()
        if teacher_model is not None:
            del teacher_model
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def set_parameters(self, base):
        """
        从服务器接收全局base模型参数
        """
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
