import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from FLAlgorithms.users.userrep import UserRep


class UserFedRepBDPFL(UserRep):
    """FedRep-compatible BDPFL submodules with phase-wise dual-path distillation."""

    def __init__(
        self,
        device,
        numeric_id,
        train_data,
        test_data,
        model,
        dataset,
        batch_size,
        learning_rate,
        beta,
        lamda,
        local_epochs,
        lr_head=None,
        plocal_epochs=1,
        malclient_ids=None,
        bd_lambda=1.0,
        bd_tau=1.0,
        bd_gamma=1.0,
        bd_use_inter=1,
        bd_use_em=1,
    ):
        super().__init__(
            device,
            numeric_id,
            train_data,
            test_data,
            model,
            dataset,
            batch_size,
            learning_rate,
            beta,
            lamda,
            local_epochs,
            lr_head=lr_head,
            plocal_epochs=plocal_epochs,
        )
        self.malclient_ids = malclient_ids if malclient_ids is not None else set()
        self.bd_lambda = float(bd_lambda)
        self.bd_tau = float(bd_tau)
        self.bd_gamma = float(bd_gamma)
        self.bd_use_inter = bool(bd_use_inter)
        self.bd_use_em = bool(bd_use_em)
        # Keep a lagged local teacher to avoid dual-branch collapse in inter-only mode.
        self.teacher_momentum = 0.9
        self.personalized_model_module = copy.deepcopy(self.model).to(self.device)
        self.optimizer_personal_base = torch.optim.SGD(self.personalized_model_module.base.parameters(), lr=self.learning_rate)
        self.optimizer_personal_head = torch.optim.SGD(self.personalized_model_module.head.parameters(), lr=self.lr_head)
        self.personalized_on_cpu = False
        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def _sync_personalized_bar(self):
        for dst, src in zip(self.persionalized_model_bar, self.personalized_model_module.parameters()):
            dst.data = src.data.clone()

    def _ensure_personalized_on_device(self):
        if self.personalized_on_cpu:
            self.personalized_model_module = self.personalized_model_module.to(self.device)
            self.personalized_on_cpu = False

    def _offload_personalized_to_cpu(self):
        if not self.personalized_on_cpu:
            self.personalized_model_module = self.personalized_model_module.to("cpu")
            self.personalized_on_cpu = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_parameters(self, model):
        super().set_parameters(model)
        self._ensure_personalized_on_device()
        if not self._is_benign():
            self.personalized_model_module.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self._sync_personalized_bar()
        self._offload_personalized_to_cpu()

    def _update_personalized_teacher(self, momentum=None):
        if momentum is None:
            momentum = self.teacher_momentum
        m = float(momentum)
        m = min(max(m, 0.0), 0.9999)
        one_minus_m = 1.0 - m
        for t_param, s_param in zip(self.personalized_model_module.parameters(), self.model.parameters()):
            t_param.data.mul_(m).add_(s_param.data, alpha=one_minus_m)

    def _feature_tap_indices(self, model):
        if len(model.base) >= 9 and isinstance(model.base[7], nn.AvgPool2d):
            return {3, 4, 5, 6}
        return {2, 5}

    def _forward_with_features(self, model, x):
        taps = self._feature_tap_indices(model)
        feats = []
        out = x
        for idx, layer in enumerate(model.base):
            out = layer(out)
            if idx in taps and out.dim() == 4:
                feats.append(out)
        logits = model.head(out)
        return logits, feats

    def _soft_kl(self, student_logits, teacher_logits):
        tau = max(self.bd_tau, 1e-6)
        student_log_prob = F.log_softmax(student_logits / tau, dim=1)
        teacher_prob = F.softmax(teacher_logits.detach() / tau, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (tau * tau)

    def _layer_weight(self, layer_idx):
        return 1.0 / (1.0 + self.bd_gamma * float(layer_idx + 1))

    def _feature_distill_loss(self, student_feats, teacher_feats, weighted):
        loss = 0.0
        for idx, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            coeff = self._layer_weight(idx) if weighted else 1.0
            loss = loss + coeff * F.mse_loss(s_feat, t_feat.detach())
        return loss

    def _gradcam_heatmaps(self, logits, feats, labels, create_graph):
        target_score = logits.gather(1, labels.view(-1, 1)).sum()
        heatmaps = []
        for feat in feats:
            grads = torch.autograd.grad(
                target_score,
                feat,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            heatmap = F.relu((weights * feat).sum(dim=1))
            norm = heatmap.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            heatmaps.append(heatmap.flatten(start_dim=1) / norm)
        return heatmaps

    def _heatmap_loss(self, student_logits, student_feats, teacher_logits, teacher_feats, labels, weighted):
        student_maps = self._gradcam_heatmaps(student_logits, student_feats, labels, create_graph=True)
        teacher_maps = self._gradcam_heatmaps(teacher_logits, teacher_feats, labels, create_graph=False)
        loss = 0.0
        for idx, (s_map, t_map) in enumerate(zip(student_maps, teacher_maps)):
            coeff = self._layer_weight(idx) if weighted else 1.0
            loss = loss + coeff * F.mse_loss(s_map, t_map.detach())
        return loss

    def _freeze_comm_head_phase(self):
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def _freeze_comm_base_phase(self):
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

    def _freeze_personal_head_phase(self):
        for param in self.personalized_model_module.base.parameters():
            param.requires_grad = False
        for param in self.personalized_model_module.head.parameters():
            param.requires_grad = True

    def _freeze_personal_base_phase(self):
        for param in self.personalized_model_module.base.parameters():
            param.requires_grad = True
        for param in self.personalized_model_module.head.parameters():
            param.requires_grad = False

    def _freeze_personal_teacher(self):
        for param in self.personalized_model_module.parameters():
            param.requires_grad = False

    def _train_head_phase_benign(self):
        self._freeze_comm_head_phase()
        self._freeze_personal_teacher()
        for _ in range(self.plocal_epochs):
            x, y = self.get_next_train_batch()
            comm_logits = self.model(x)
            with torch.no_grad():
                personal_logits = self.personalized_model_module(x)
            comm_loss = self.loss(comm_logits, y) + self.bd_lambda * self._soft_kl(comm_logits, personal_logits)
            self.optimizer_head.zero_grad()
            comm_loss.backward()
            self.optimizer_head.step()

    def _train_base_phase_benign(self):
        self._freeze_comm_base_phase()
        self._freeze_personal_teacher()
        for _ in range(self.local_epochs):
            x, y = self.get_next_train_batch()
            comm_logits, comm_feats = self._forward_with_features(self.model, x)
            if self.bd_use_em:
                # Grad-CAM teacher maps need logits->feature autograd graph.
                x_teacher = x.detach().requires_grad_(True)
                personal_logits, personal_feats = self._forward_with_features(self.personalized_model_module, x_teacher)
            else:
                with torch.no_grad():
                    personal_logits, personal_feats = self._forward_with_features(self.personalized_model_module, x)

            comm_loss = self.loss(comm_logits, y) + self.bd_lambda * self._soft_kl(comm_logits, personal_logits)

            if self.bd_use_inter:
                comm_loss = comm_loss + self._feature_distill_loss(comm_feats, personal_feats, weighted=True)

            if self.bd_use_em:
                comm_loss = comm_loss + self._heatmap_loss(
                    comm_logits, comm_feats, personal_logits, personal_feats, y, weighted=True
                )
            self.optimizer_base.zero_grad()
            comm_loss.backward()
            self.optimizer_base.step()

    def train(self):
        self.model.train()
        self._ensure_personalized_on_device()
        self.personalized_model_module.train()

        if not self._is_benign():
            result = super().train()
            self.personalized_model_module.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self._sync_personalized_bar()
            self._offload_personalized_to_cpu()
            return result

        self._train_head_phase_benign()
        self._train_base_phase_benign()
        self._update_personalized_teacher()
        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()
        return 0

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        result = super().poison_all_train(poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate)
        self._ensure_personalized_on_device()
        self.personalized_model_module.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()
        return result

    def test_persionalized_model(self):
        self._ensure_personalized_on_device()
        origin_model = self.model
        self.model = self.personalized_model_module
        try:
            result = super().test()
        finally:
            self.model = origin_model
            self._offload_personalized_to_cpu()
        return result

    def train_error_and_loss_persionalized_model(self):
        self._ensure_personalized_on_device()
        origin_model = self.model
        self.model = self.personalized_model_module
        try:
            result = super().train_error_and_loss()
        finally:
            self.model = origin_model
            self._offload_personalized_to_cpu()
        return result

    def poisontest_persionalized_model(self, poiosnlabel, trigger, pattern):
        self._ensure_personalized_on_device()
        origin_model = self.model
        self.model = self.personalized_model_module
        try:
            result = super().poisontest(poiosnlabel, trigger, pattern)
        finally:
            self.model = origin_model
            self._offload_personalized_to_cpu()
        return result

    def poison_train_error_and_loss_persionalized_model(self, poiosnlabel, trigger, pattern):
        self._ensure_personalized_on_device()
        origin_model = self.model
        self.model = self.personalized_model_module
        try:
            result = super().poison_train_error_and_loss(poiosnlabel=poiosnlabel, trigger=trigger, pattern=pattern)
        finally:
            self.model = origin_model
            self._offload_personalized_to_cpu()
        return result
