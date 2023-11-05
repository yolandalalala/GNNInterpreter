from torch import nn


class NormPenalty(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.norm(p=self.order)

