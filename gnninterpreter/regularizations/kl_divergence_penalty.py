import torch

from . import BaseRegularization


# TODO: with logits
class KLDivergencePenalty(BaseRegularization):
    def __init__(self, getter, weight=1, binary=True, mean=True, eps=1e-4, **kwargs):
        super().__init__(getter, weight, mean=mean, **kwargs)
        self.binary = binary
        self.eps = eps

    def penalty(self, p, q):
        p = p * (1 - 2*self.eps) + self.eps
        q = q * (1 - 2*self.eps) + self.eps
        if self.binary:
            p = torch.stack([p, 1-p], dim=-1)
            q = torch.stack([q, 1-q], dim=-1)
        return torch.sum(p * (p / q).log())


