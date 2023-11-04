import torch
from torch import nn
import torch.nn.functional as F


class WeightedCriterion(nn.Module):
    def __init__(self, criteria):
        super().__init__()
        self.criteria = criteria

    def forward(self, x):
        loss = 0
        for criterion in self.criteria:
            loss += criterion["criterion"](x[criterion["key"]]) * criterion["weight"]
        return loss

