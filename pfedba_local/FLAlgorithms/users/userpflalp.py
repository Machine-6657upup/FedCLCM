"""
Paper-aligned PFL-ALP full-model client.
"""
import copy
import math

import torch
import torch.nn.functional as F

from FLAlgorithms.users.useravg import UserAVG


class UserPFLALP(UserAVG):
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
        purify_beta=1500.0,
        purify_rounds=1,
        mal_local_epochs=None,
        mal_learning_rate=None,
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
        self.purify_beta = float(purify_beta)
        self.purify_rounds = max(int(purify_rounds), 1)
        self.mal_local_epochs = max(int(mal_local_epochs if mal_local_epochs is not None else local_epochs), 1)
        self.mal_learning_rate = self.learning_rate if mal_learning_rate is None else float(mal_learning_rate)
        self.personalized_model_module = None
        self.personalized_on_cpu = False

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def set_learning_rate(self, lr):
        self.learning_rate = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _clone_model(self, model):
        cloned = copy.deepcopy(model).to(self.device)
        cloned.train()
        return cloned

    def _ensure_personalized_on_device(self):
        if self.personalized_model_module is not None and self.personalized_on_cpu:
            self.personalized_model_module = self.personalized_model_module.to(self.device)
            self.personalized_on_cpu = False

    def _offload_personalized_to_cpu(self):
        if self.personalized_model_module is not None and not self.personalized_on_cpu:
            self.personalized_model_module = self.personalized_model_module.to("cpu")
            self.personalized_on_cpu = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _sync_personalized_bar(self, source_model=None):
        model = source_model if source_model is not None else self.personalized_model_module
        if model is None:
            model = self.model
        for dst, src in zip(self.persionalized_model_bar, model.parameters()):
            dst.data = src.data.detach().to(dst.device).clone()

    def set_parameters(self, model):
        super().set_parameters(model)
        if (not self._is_benign()) or (self.personalized_model_module is None):
            self._sync_personalized_bar(self.model)

    def _forward_with_features(self, model, x):
        if all(hasattr(model, key) for key in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear"]):
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

        if all(hasattr(model, key) for key in ["conv1", "conv2", "fc1", "fc2"]):
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

        raise ValueError("PFL-ALP currently supports PFedBA CIFAR/MNIST style models only.")

    def _nad_loss(self, student_feat, teacher_feat, power=2):
        def _attention_map(feat):
            attn = feat.abs().pow(power).mean(dim=1)
            norm = torch.norm(attn.flatten(start_dim=1), dim=1, keepdim=True).clamp_min(1e-12)
            return attn.flatten(start_dim=1) / norm

        student_attention = _attention_map(student_feat)
        teacher_attention = _attention_map(teacher_feat.detach())
        return F.mse_loss(student_attention, teacher_attention)

    def _update_model_on_main_task(self, model, learning_rate=None):
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train()
        for datas, labels in self.trainloader:
            x, y = datas.to(self.device), labels.to(self.device)
            logits = model(x)
            loss = self.loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train(self):
        result = super().train()
        if (not self._is_benign()) or (self.personalized_model_module is None):
            self._sync_personalized_bar(self.model)
        return result

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        last_local_model = {}
        for name, data in self.model.state_dict().items():
            last_local_model[name] = data.clone()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.mal_learning_rate)
        self.model.train()
        for _ in range(1, self.mal_local_epochs + 1):
            x, y = self.get_next_poison_all_train_batch(
                poison_ratio=poison_ratio,
                poison_label=poison_label,
                noise_trigger=trigger,
                pattern=pattern,
            )
            optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            optimizer.step()

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
            print("scaled distance:{}".format(math.sqrt(squared_sum)))

        self.personalized_model_module = None
        self._sync_personalized_bar(self.model)
        return 0

    def personalize_from_representative(self, representative_model):
        if representative_model is None:
            self._sync_personalized_bar(self.model)
            return

        if not self._is_benign():
            self.personalized_model_module = None
            self._sync_personalized_bar(self.model)
            return

        teacher_source = self.personalized_model_module if self.personalized_model_module is not None else self.model
        teacher_model = self._clone_model(teacher_source)
        student_model = self._clone_model(representative_model)

        for _ in range(self.purify_rounds):
            self._update_model_on_main_task(teacher_model)
            teacher_model.eval()

            optimizer = torch.optim.SGD(student_model.parameters(), lr=self.learning_rate)
            student_model.train()
            for datas, labels in self.trainloader:
                x, y = datas.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    _, teacher_feats = self._forward_with_features(teacher_model, x)
                student_logits, student_feats = self._forward_with_features(student_model, x)
                loss = self.loss(student_logits, y)
                if self.purify_beta > 0:
                    nad_loss = 0.0
                    for student_feat, teacher_feat in zip(student_feats, teacher_feats):
                        nad_loss = nad_loss + self._nad_loss(student_feat, teacher_feat)
                    loss = loss + self.purify_beta * nad_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.personalized_model_module = self._clone_model(student_model)
        self._sync_personalized_bar(self.personalized_model_module)
        self._offload_personalized_to_cpu()
