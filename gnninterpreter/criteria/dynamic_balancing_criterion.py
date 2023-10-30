import torch
from torch import nn
import torch.nn.functional as F


class DynamicBalancingCriterion(nn.Module):
    def __init__(self, classes, alpha=1, beta=1):
        super().__init__()
        self.classes = classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits):
        assert len(logits.shape) == 2
        probs = F.softmax(logits, dim=-1).detach()
        mask = torch.zeros_like(logits).bool()
        mask[:, self.classes] = True
        score = logits * probs ** 2
        score[mask] = logits[mask] * (1 - probs[mask]) ** 2
        return (score[~mask].mean(dim=-1) - score[mask].mean(dim=-1)).mean()
