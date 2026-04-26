import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        return self.head(self.base(x))


class LogSoftmaxHead(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


class LinearHead(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)


def to_fedrep_model(model):
    # MnistNet / FMnistNet style
    if all(hasattr(model, k) for k in ["conv1", "conv2", "fc1", "fc2"]):
        base = nn.Sequential(
            model.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            model.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            model.fc1,
            nn.ReLU(),
        )
        head = LogSoftmaxHead(model.fc2)
        return BaseHeadSplit(base, head)
    # ResNet (CIFAR) style
    if all(hasattr(model, k) for k in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear"]):
        base = nn.Sequential(
            model.conv1,
            model.bn1,
            nn.ReLU(),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AvgPool2d(kernel_size=4),
            Flatten(),
        )
        head = LinearHead(model.linear)
        return BaseHeadSplit(base, head)
    raise ValueError("FedRep split currently supports models with conv1/conv2/fc1/fc2.")
