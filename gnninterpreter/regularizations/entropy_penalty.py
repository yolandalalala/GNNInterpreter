import torch
from torch import nn


# TODO: with logits
class EntropyPenalty(nn.Module):
    def __init__(self, binary=True, eps=1e-4):
        super().__init__()
        self.binary = binary
        self.eps = eps

    def forward(self, p):
        p = p * (1 - 2*self.eps) + self.eps
        if self.binary:
            p = torch.stack([p, 1-p], dim=-1)
        return -torch.sum(p * p.log())

