"""
FedRPD client for PFedBA:
- starts from the current FedRT local routine
- adds benign-side local purification inspired by PFL-ALP
- adds layer-wise feature distillation inspired by BDPFL

This is intentionally a paper-grounded lite adaptation:
- no official public source code was confirmed for BDPFL / PFL-ALP
- defaults are kept few and conservative
"""
import copy
import time

import torch
import torch.nn.functional as F

from FLAlgorithms.users.userfedrt import UserFedRT


class UserFedRPD(UserFedRT):
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
        aug_strength=0.0,
        adv_eps=0.0,
        adv_num_iter=0,
        purify_beta=1500.0,
        purify_rounds=1,
        distill_gamma=1.0,
        distill_weight=1.0,
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
            malclient_ids=malclient_ids,
            aug_strength=aug_strength,
            adv_eps=adv_eps,
            adv_num_iter=adv_num_iter,
        )
        self.purify_beta = float(purify_beta)
        self.purify_rounds = max(int(purify_rounds), 1)
        self.distill_gamma = float(distill_gamma)
        self.distill_weight = float(distill_weight)
        self.personal_teacher = copy.deepcopy(self.model).to(self.device)
        for param in self.personal_teacher.parameters():
            param.requires_grad = False

    def _sync_teacher_from_student(self):
        self.personal_teacher.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.personal_teacher.eval()
        for param in self.personal_teacher.parameters():
            param.requires_grad = False

    def _forward_with_features(self, model, x):
        features = []
        out = x
        base = model.base
        # ResNet FedRep split: conv1, bn1, relu, layer1, layer2, layer3, layer4, avgpool, flatten
        for idx, layer in enumerate(base):
            out = layer(out)
            if idx in (3, 4, 5, 6):
                features.append(out)
        logits = model.head(out)
        return logits, features

    def _nad_loss(self, student_feat, teacher_feat, power=2):
        def _attn(feat):
            attn = feat.abs().pow(power).mean(dim=1)
            norm = torch.norm(attn.flatten(start_dim=1), dim=1, keepdim=True).clamp_min(1e-12)
            return attn.flatten(start_dim=1) / norm

        s = _attn(student_feat)
        t = _attn(teacher_feat.detach())
        return F.mse_loss(s, t)

    def _layerwise_feature_loss(self, student_feats, teacher_feats):
        loss = 0.0
        for layer_idx, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            weight = 1.0 / (1.0 + self.distill_gamma * layer_idx)
            loss = loss + weight * F.mse_loss(s_feat, t_feat.detach())
        return loss

    def _purify_benign_model(self):
        if self.purify_beta <= 0 and self.distill_weight <= 0:
            self._sync_teacher_from_student()
            return

        self.personal_teacher.eval()
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for _ in range(self.purify_rounds):
            x, y = self.get_next_train_batch()
            x_in = self.trigger_breaking_augment(x)

            with torch.no_grad():
                _, teacher_feats = self._forward_with_features(self.personal_teacher, x_in)

            student_logits, student_feats = self._forward_with_features(self.model, x_in)
            loss = self.loss(student_logits, y)

            if self.purify_beta > 0:
                nad = 0.0
                for s_feat, t_feat in zip(student_feats, teacher_feats):
                    nad = nad + self._nad_loss(s_feat, t_feat)
                loss = loss + self.purify_beta * nad

            if self.distill_weight > 0:
                feat_loss = self._layerwise_feature_loss(student_feats, teacher_feats)
                loss = loss + self.distill_weight * feat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._sync_teacher_from_student()

    def train(self):
        start_time = time.time()
        self.model.train()
        is_benign = self._is_benign()
        enable_pgd_head = is_benign and (self.adv_eps > 0) and (self.adv_num_iter > 0)

        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        if enable_pgd_head:
            for _ in range(self.plocal_epochs):
                x, y = self.get_next_train_batch()
                x_base = self.trigger_breaking_augment(x)
                x_adv = self.pgd_attack(x_base, y)
                out = self.model(x_adv)
                loss = self.loss(out, y)
                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_head.step()

        for _ in range(self.plocal_epochs):
            x, y = self.get_next_train_batch()
            out = self.model(x)
            loss = self.loss(out, y)
            self.optimizer_head.zero_grad()
            loss.backward()
            self.optimizer_head.step()

        for p in self.model.base.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = False

        for _ in range(self.local_epochs):
            x, y = self.get_next_train_batch()
            x_main = self.trigger_breaking_augment(x) if is_benign else x
            out = self.model(x_main)
            loss = self.loss(out, y)
            self.optimizer_base.zero_grad()
            loss.backward()
            self.optimizer_base.step()

        for p in self.model.parameters():
            p.requires_grad = True

        if is_benign:
            self._purify_benign_model()
        else:
            self._sync_teacher_from_student()

        return time.time() - start_time
