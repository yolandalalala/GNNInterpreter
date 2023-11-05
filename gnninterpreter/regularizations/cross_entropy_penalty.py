import torch
from torch import nn


# TODO: with logits
class CrossEntropyPenalty(nn.Module):
    def __init__(self, binary=True, eps=1e-4):
        super().__init__()
        self.binary = binary
        self.eps = eps

    def forward(self, pq):
        p = pq[0] * (1 - 2*self.eps) + self.eps
        q = pq[1] * (1 - 2*self.eps) + self.eps
        if self.binary:
            p = torch.stack([p, 1-p], dim=-1)
            q = torch.stack([q, 1-q], dim=-1)
        return -torch.sum(p * q.log())

