import torch
from torch import nn
import torch.nn.functional as F


class PerplexityCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        assert len(logits.shape) == 2
        return torch.exp(-F.log_softmax(logits, dim=-1).mean(dim=-1))
