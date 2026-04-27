"""
FedCLCMPurify client for the PFedBA stack.

This keeps the existing PFedBA FedCLCM path intact and adds only a benign-client
EMA teacher with layer attention alignment during base training.
"""
import copy

import torch
import torch.nn.functional as F

from FLAlgorithms.users.userclcm import UserCLCM


class UserCLCMPurify(UserCLCM):
    def __init__(
        self,
        *args,
        purify_beta=0.0,
        purify_feature_beta=0.0,
        purify_logit_beta=0.0,
        purify_temperature=2.0,
        purify_start_round=1,
        purify_layers="layer4",
        purify_teacher_momentum=0.9,
        purify_teacher_cpu_half=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.purify_beta = float(purify_beta)
        self.purify_feature_beta = float(purify_feature_beta)
        self.purify_logit_beta = float(purify_logit_beta)
        self.purify_temperature = float(purify_temperature)
        self.purify_start_round = int(purify_start_round)
        self.purify_teacher_momentum = float(purify_teacher_momentum)
        self.purify_teacher_cpu_half = bool(purify_teacher_cpu_half)
        self.purify_layers = [s.strip() for s in str(purify_layers).split(",") if s.strip()]
        self.purify_teacher_state = None
        self._round_idx = 0

    def _pack_teacher_state(self):
        state = {}
        for name, value in self.model.state_dict().items():
            tensor = value.detach().cpu().clone()
            if self.purify_teacher_cpu_half and torch.is_floating_point(tensor):
                tensor = tensor.half()
            state[name] = tensor
        return state

    def _update_purify_teacher(self):
        if not self._is_benign():
            return
        new_state = self._pack_teacher_state()
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
                mixed = old_tensor.float().mul(momentum).add(new_tensor.float(), alpha=1.0 - momentum)
                if self.purify_teacher_cpu_half:
                    mixed = mixed.half()
                mixed_state[name] = mixed
            else:
                mixed_state[name] = new_tensor
        self.purify_teacher_state = mixed_state

    def _build_purify_teacher(self):
        if not (
            self._is_benign()
            and self.purify_teacher_state is not None
            and self._round_idx >= self.purify_start_round
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

    def _resolve_layer_name(self, base, name):
        modules = dict(base.named_modules())
        if name in modules:
            return name
        # PFedBA's FedRep split wraps CIFAR ResNet layer4 as Sequential index 6.
        aliases = {
            "layer1": "3",
            "layer2": "4",
            "layer3": "5",
            "layer4": "6",
        }
        alias = aliases.get(name)
        if alias in modules:
            return alias
        return None

    def _forward_base_with_activations(self, base, x, eval_mode=False):
        modules = dict(base.named_modules())
        activations = {}
        handles = []
        was_training = base.training

        def make_hook(name):
            def hook(_module, _inputs, output):
                activations[name] = output
            return hook

        for raw_name in self.purify_layers:
            name = self._resolve_layer_name(base, raw_name)
            if name is not None and name in modules:
                handles.append(modules[name].register_forward_hook(make_hook(raw_name)))

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

    def train(self):
        self.model.train()
        is_benign = self._is_benign()
        teacher_model = self._build_purify_teacher()
        enable_pgd_head = is_benign and (self.adv_eps > 0) and (self.adv_num_iter > 0)

        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        for _ in range(self.plocal_epochs):
            for _, (datas, labels) in enumerate(self.trainloader):
                x, y = datas.to(self.device), labels.to(self.device)
                x_base = self.trigger_breaking_augment(x) if is_benign else x
                x_in = self.pgd_attack(x_base, y) if enable_pgd_head else x_base
                out = self.model(x_in)
                loss = self.loss(out, y)
                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_head.step()

        for p in self.model.base.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = False

        for _ in range(self.local_epochs):
            for _, (datas, labels) in enumerate(self.trainloader):
                x, y = datas.to(self.device), labels.to(self.device)
                x_main = self.trigger_breaking_augment(x) if is_benign else x
                if is_benign:
                    x_aug1, x_aug2 = self.generate_augmented_views(x_main)
                    h1 = self.model.base(x_aug1)
                    h2 = self.model.base(x_aug2)
                    cl_loss = self.contrastive_loss(h1, h2)
                    out = self.model(x_main)
                    ce_loss = self.loss(out, y)
                    loss = ce_loss + self.lambda_cl * cl_loss
                    if teacher_model is not None:
                        loss = loss + self.purification_loss(x, teacher_model)
                else:
                    out = self.model(x)
                    loss = self.loss(out, y)
                self.optimizer_base.zero_grad()
                loss.backward()
                self.optimizer_base.step()

        for p in self.model.parameters():
            p.requires_grad = True

        if is_benign:
            self._update_purify_teacher()
        if teacher_model is not None:
            del teacher_model
        self._round_idx += 1
        return 0
