from . import BaseRegularization
import torch.nn.functional as F

class BudgetPenalty(BaseRegularization):
    def __init__(self, getter, budget=0, order=1, beta=1, weight=1, mean=True, **kwargs):
        super().__init__(getter, weight, mean, **kwargs)
        self.budget = budget
        self.beta = beta
        self.order = order

    def penalty(self, x):
        # compute budget deficit
        return F.softplus(x.sum() - self.budget, beta=self.beta) ** self.order

