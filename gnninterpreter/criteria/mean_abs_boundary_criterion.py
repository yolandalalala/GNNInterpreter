import torch
from torch import nn
import torch.nn.functional as F


class MeanAbsBoundaryCriterion(nn.Module):
    def __init__(self, class_a, class_b):
        super().__init__()
        self.class_a = class_a
        self.class_b = class_b

    def forward(self, logits):
        return (logits[:, self.class_a] - logits[:, self.class_b]).abs().mean()
