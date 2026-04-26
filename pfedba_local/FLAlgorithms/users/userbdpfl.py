"""
BDPFL full-style client for the official PFedBA workspace.
Communication model is uploaded to server; personalized model stays local.
"""
import copy
import time

import torch
import torch.nn.functional as F

from FLAlgorithms.users.useravg import UserAVG


class UserBDPFL(UserAVG):
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
        )
        self.malclient_ids = malclient_ids if malclient_ids is not None else set()
        self.bd_lambda = float(bd_lambda)
        self.bd_tau = float(bd_tau)
        self.bd_gamma = float(bd_gamma)
        self.bd_use_inter = bool(bd_use_inter)
        self.bd_use_em = bool(bd_use_em)
        self.personalized_model_module = copy.deepcopy(self.model).to(self.device)
        self.optimizer_personal = torch.optim.SGD(self.personalized_model_module.parameters(), lr=self.learning_rate)
        self.personalized_on_cpu = False
        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def set_learning_rate(self, lr):
        self.learning_rate = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.optimizer_personal.param_groups:
            group["lr"] = self.learning_rate

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

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

    def _forward_with_features(self, model, x):
        if all(hasattr(model, k) for k in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear"]):
            feats = []
            out = F.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            feats.append(out)
            out = model.layer2(out)
            feats.append(out)
            out = model.layer3(out)
            feats.append(out)
            out = model.layer4(out)
            feats.append(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            logits = model.linear(out)
            return logits, feats

        if all(hasattr(model, k) for k in ["conv1", "conv2", "fc1", "fc2"]):
            feats = []
            out = F.relu(model.conv1(x))
            feats.append(out)
            out = F.max_pool2d(out, 2, 2)
            out = F.relu(model.conv2(out))
            feats.append(out)
            out = F.max_pool2d(out, 2, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(model.fc1(out))
            logits = model.fc2(out)
            return logits, feats

        raise ValueError("BDPFL currently supports PFedBA CIFAR/MNIST style models only.")

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
        last_idx = len(feats) - 1
        for idx, feat in enumerate(feats):
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

    def train(self):
        start_time = time.time()
        self.model.train()
        self._ensure_personalized_on_device()
        self.personalized_model_module.train()

        if not self._is_benign():
            for epoch in range(1, self.local_epochs + 1):
                x, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
            self.personalized_model_module.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self._sync_personalized_bar()
            self._offload_personalized_to_cpu()
            return time.time() - start_time

        for _ in range(self.local_epochs):
            x, y = self.get_next_train_batch()

            comm_logits, comm_feats = self._forward_with_features(self.model, x)
            personal_logits, personal_feats = self._forward_with_features(self.personalized_model_module, x)

            comm_loss = self.loss(comm_logits, y) + self.bd_lambda * self._soft_kl(comm_logits, personal_logits)
            personal_loss = self.loss(personal_logits, y) + self.bd_lambda * self._soft_kl(personal_logits, comm_logits)

            if self.bd_use_inter:
                comm_loss = comm_loss + self._feature_distill_loss(comm_feats, personal_feats, weighted=True)
                personal_loss = personal_loss + self._feature_distill_loss(personal_feats, comm_feats, weighted=False)

            if self.bd_use_em:
                comm_loss = comm_loss + self._heatmap_loss(
                    comm_logits, comm_feats, personal_logits, personal_feats, y, weighted=True
                )
                # Recompute for the reverse direction so the first Grad-CAM call does not
                # consume the graph that the second direction still needs.
                personal_logits_em, personal_feats_em = self._forward_with_features(self.personalized_model_module, x)
                comm_logits_em, comm_feats_em = self._forward_with_features(self.model, x)
                personal_loss = personal_loss + self._heatmap_loss(
                    personal_logits_em, personal_feats_em, comm_logits_em, comm_feats_em, y, weighted=False
                )

            total_loss = comm_loss + personal_loss
            self.optimizer.zero_grad()
            self.optimizer_personal.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer_personal.step()

        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()
        return time.time() - start_time

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        loss = 0
        last_local_model = {}
        for name, data in self.model.state_dict().items():
            last_local_model[name] = data.clone()

        self.model.train()
        for _ in range(1, self.local_epochs + 1):
            x, y = self.get_next_poison_all_train_batch(
                poison_ratio=poison_ratio,
                poison_label=poison_label,
                noise_trigger=trigger,
                pattern=pattern,
            )
            self.optimizer.zero_grad()
            output = self.model(x)
            ce_loss = self.loss(output, y)
            ce_loss.backward()
            self.optimizer.step()

        if oneshot == 1:
            now_local_model = {}
            for name, data in self.model.state_dict().items():
                now_local_model[name] = data.clone()
            print("scale the local model update!")
            for key, value in self.model.state_dict().items():
                target_value = last_local_model[key]
                new_value = target_value + (value - target_value) * clip_rate
                self.model.state_dict()[key].copy_(new_value)
            squared_sum = 0
            for name, layer in self.model.named_parameters():
                squared_sum += torch.sum(torch.pow(layer.data - now_local_model[name].data, 2))
            print("scaled distance:{}".format((squared_sum.sqrt()).item()))

        self.personalized_model_module.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self._sync_personalized_bar()
        self._offload_personalized_to_cpu()
        return loss

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
