import torch.nn.functional as F
from torch import nn


class BudgetPenalty(nn.Module):
    def __init__(self, budget=0, order=1, beta=1):
        super().__init__()
        self.budget = budget
        self.beta = beta
        self.order = order

    def forward(self, theta):
        return F.softplus(theta.sum() - self.budget, beta=self.beta) ** self.order

