import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from FLAlgorithms.users.userrep import UserRep


class UserFedRepPFLALP(UserRep):
    """FedRep-compatible PFL-ALP submodules: representative clustering + NAD purification."""

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
        alp_use_purify=1,
        purify_beta=1500.0,
        purify_rounds=1,
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
        self.alp_use_purify = bool(alp_use_purify)
        self.purify_beta = float(purify_beta)
        self.purify_rounds = max(int(purify_rounds), 1)
        self.personalized_model_module = None

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def _clone_model(self, model):
        cloned = copy.deepcopy(model).to(self.device)
        cloned.train()
        return cloned

    def _sync_personalized_bar(self, source_model=None):
        model = source_model if source_model is not None else self.personalized_model_module
        if model is None:
            model = self.model
        for dst, src in zip(self.persionalized_model_bar, model.parameters()):
            dst.data = src.data.clone()

    def set_parameters(self, model):
        super().set_parameters(model)
        if (not self._is_benign()) or (self.personalized_model_module is None):
            self._sync_personalized_bar(self.model)

    def apply_personalized_initializer(self):
        if (not self._is_benign()) or (self.personalized_model_module is None):
            return
        self.model.load_state_dict(copy.deepcopy(self.personalized_model_module.state_dict()))
        for local_param, new_param in zip(self.local_model, self.model.parameters()):
            local_param.data = new_param.data.clone()

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

    def _nad_loss(self, student_feat, teacher_feat, power=2):
        def _attention_map(feat):
            attn = feat.abs().pow(power).mean(dim=1)
            norm = torch.norm(attn.flatten(start_dim=1), dim=1, keepdim=True).clamp_min(1e-12)
            return attn.flatten(start_dim=1) / norm

        return F.mse_loss(_attention_map(student_feat), _attention_map(teacher_feat.detach()))

    def _build_personalized_student(self, representative_model):
        student = self._clone_model(representative_model)
        for student_param, local_param in zip(student.head.parameters(), self.model.head.parameters()):
            student_param.data = local_param.data.clone()
        return student

    def train(self):
        result = super().train()
        if (not self._is_benign()) or (self.personalized_model_module is None):
            self._sync_personalized_bar(self.model)
        return result

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        result = super().poison_all_train(poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate)
        self.personalized_model_module = None
        self._sync_personalized_bar(self.model)
        return result

    def personalize_with_representative(self, representative_model):
        if representative_model is None:
            self._sync_personalized_bar(self.model)
            return

        if not self._is_benign():
            self.personalized_model_module = None
            self._sync_personalized_bar(self.model)
            return

        student_model = self._build_personalized_student(representative_model)

        if not self.alp_use_purify:
            self.personalized_model_module = student_model
            self._sync_personalized_bar(student_model)
            return

        teacher_model = self._clone_model(self.model)
        teacher_model.eval()
        optimizer = torch.optim.SGD(student_model.parameters(), lr=self.learning_rate)

        for _ in range(self.purify_rounds):
            student_model.train()
            for datas, labels in self.trainloader:
                x, y = datas.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    _, teacher_feats = self._forward_with_features(teacher_model, x)
                student_logits, student_feats = self._forward_with_features(student_model, x)
                loss = self.loss(student_logits, y)
                if self.purify_beta > 0:
                    nad_loss = 0.0
                    for s_feat, t_feat in zip(student_feats, teacher_feats):
                        nad_loss = nad_loss + self._nad_loss(s_feat, t_feat)
                    loss = loss + self.purify_beta * nad_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.personalized_model_module = student_model
        self._sync_personalized_bar(student_model)
