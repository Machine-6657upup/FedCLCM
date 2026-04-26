"""
PFL-ALP-full style user for the official PFedBA workspace.
"""
import copy

import torch
import torch.nn.functional as F

from FLAlgorithms.users.userrep import UserRep


class UserPFLALP(UserRep):
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
        self.purify_beta = float(purify_beta)
        self.purify_rounds = max(int(purify_rounds), 1)
        self.personalized_model_module = None

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def _clone_model(self, model):
        cloned = copy.deepcopy(model).to(self.device)
        cloned.train()
        return cloned

    def _sync_personalized_bar(self):
        if self.personalized_model_module is None:
            return
        for dst, src in zip(self.persionalized_model_bar, self.personalized_model_module.parameters()):
            dst.data = src.data.clone()

    def set_parameters(self, model):
        super().set_parameters(model)
        if (not self._is_benign()) or (self.personalized_model_module is None):
            for dst, src in zip(self.persionalized_model_bar, self.model.parameters()):
                dst.data = src.data.clone()

    def _forward_with_features(self, model, x):
        features = []
        out = x
        for idx, layer in enumerate(model.base):
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

    def _update_model_on_main_task(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()
        for _, (datas, labels) in enumerate(self.trainloader):
            x, y = datas.to(self.device), labels.to(self.device)
            logits = model(x)
            loss = self.loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def personalize_from_representative(self, representative_model):
        if not self._is_benign():
            return

        if self.personalized_model_module is None:
            self.personalized_model_module = self._clone_model(self.model)

        teacher_model = self._clone_model(self.personalized_model_module)
        student_model = self._clone_model(representative_model)

        for _ in range(self.purify_rounds):
            self._update_model_on_main_task(teacher_model)
            teacher_model.eval()

            optimizer = torch.optim.SGD(student_model.parameters(), lr=self.learning_rate)
            student_model.train()
            for _, (datas, labels) in enumerate(self.trainloader):
                x, y = datas.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    _, teacher_feats = self._forward_with_features(teacher_model, x)
                student_logits, student_feats = self._forward_with_features(student_model, x)
                loss = self.loss(student_logits, y)
                if self.purify_beta > 0:
                    nad = 0.0
                    for s_feat, t_feat in zip(student_feats, teacher_feats):
                        nad = nad + self._nad_loss(s_feat, t_feat)
                    loss = loss + self.purify_beta * nad
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            teacher_model = self._clone_model(student_model)

        self.personalized_model_module = self._clone_model(student_model)
        self._sync_personalized_bar()
