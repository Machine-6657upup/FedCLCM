"""
FedRT client for PFedBA: FedRep-style base/head split with
- benign head PGD purification
- benign optional trigger-breaking augmentation
- shared-base local training
"""
import time

import torch
import torch.nn.functional as F

from FLAlgorithms.users.userrep import UserRep


class UserFedRT(UserRep):
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
        self.aug_strength = float(aug_strength)
        self.adv_eps = float(adv_eps)
        self.adv_num_iter = int(adv_num_iter)
        self.adv_step = (
            (2 * self.adv_eps / self.adv_num_iter) if (self.adv_num_iter > 0 and self.adv_eps > 0) else 0.0
        )

    def _is_benign(self):
        return str(self.id) not in self.malclient_ids

    def trigger_breaking_augment(self, x):
        if self.aug_strength <= 0:
            return x
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
            x[i, :, y0 : y0 + cut_h, x0 : x0 + cut_w] = torch.empty(
                (c, cut_h, cut_w), device=x.device, dtype=x.dtype
            ).uniform_(-1.0, 1.0)
        return x

    def pgd_attack(self, x, y):
        if self.adv_eps <= 0 or self.adv_num_iter <= 0:
            return x
        x_orig = x.detach()
        delta = torch.zeros_like(x_orig).uniform_(-self.adv_eps, self.adv_eps)
        adv_x = torch.clamp(x_orig + delta, -1, 1)
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

        return time.time() - start_time
