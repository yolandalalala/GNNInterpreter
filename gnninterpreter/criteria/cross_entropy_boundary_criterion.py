import torch
from torch import nn


class CrossEntropyBoundaryCriterion(nn.Module):
    def __init__(self, class_a, class_b):
        super().__init__()
        self.class_a = class_a
        self.class_b = class_b
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits):
        target = torch.zeros_like(logits)
        target[:, [self.class_a, self.class_b]] = 0.5
        return self.criterion(logits, target)
