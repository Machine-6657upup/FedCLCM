import math

import torch
import torch.nn as nn

from FLAlgorithms.users.useravg import UserAVG


class UserRep(UserAVG):
    """FedRep user: local head + shared base."""

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
        self.loss = nn.CrossEntropyLoss()
        self.plocal_epochs = max(int(plocal_epochs), 1)
        self.lr_head = self.learning_rate if lr_head is None else float(lr_head)
        self.optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.lr_head)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.base.parameters(), model.base.parameters()):
            old_param.data = new_param.data.clone()
        for local_param, new_param in zip(self.local_model, self.model.parameters()):
            local_param.data = new_param.data.clone()

    def _train_head_phase(self, poison=False, poison_ratio=0, poison_label=0, trigger=None, pattern=None):
        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        for _ in range(self.plocal_epochs):
            if poison:
                x, y = self.get_next_poison_all_train_batch(
                    poison_ratio=poison_ratio,
                    poison_label=poison_label,
                    noise_trigger=trigger,
                    pattern=pattern,
                )
            else:
                x, y = self.get_next_train_batch()
            self.optimizer_head.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer_head.step()

    def _train_base_phase(self, poison=False, poison_ratio=0, poison_label=0, trigger=None, pattern=None):
        for p in self.model.base.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = False

        for _ in range(self.local_epochs):
            if poison:
                x, y = self.get_next_poison_all_train_batch(
                    poison_ratio=poison_ratio,
                    poison_label=poison_label,
                    noise_trigger=trigger,
                    pattern=pattern,
                )
            else:
                x, y = self.get_next_train_batch()
            self.optimizer_base.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer_base.step()

    def train(self):
        self.model.train()
        self._train_head_phase(poison=False)
        self._train_base_phase(poison=False)
        return 0

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        last_local_model = {}
        for name, data in self.model.state_dict().items():
            last_local_model[name] = data.clone()

        self.model.train()
        self._train_head_phase(
            poison=True,
            poison_ratio=poison_ratio,
            poison_label=poison_label,
            trigger=trigger,
            pattern=pattern,
        )
        self._train_base_phase(
            poison=True,
            poison_ratio=poison_ratio,
            poison_label=poison_label,
            trigger=trigger,
            pattern=pattern,
        )

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

        return 0

    def train_one_step(self, per_epochs):
        # FedRep personalized fine-tuning: update head only.
        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True
        self.model.train()
        for _ in range(per_epochs):
            for _, (datas, labels) in enumerate(self.trainloader):
                x, y = datas.to(self.device), labels.to(self.device)
                self.optimizer_head.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_head.step()

    def train_one_step_poison(self, per_epochs, trigger, pattern, poison_label, poison_ratio):
        # FedRep personalized poisoned fine-tuning: update head only.
        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True
        self.model.train()
        for _ in range(per_epochs):
            for _, batch in enumerate(self.trainloader):
                x, y = self.get_poison_batch(
                    batch=batch,
                    trigger=trigger,
                    pattern=pattern,
                    poison_label=poison_label,
                    poison_ratio=poison_ratio,
                )
                self.optimizer_head.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_head.step()
