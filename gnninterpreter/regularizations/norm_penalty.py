from . import BaseRegularization


class NormPenalty(BaseRegularization):
    def __init__(self, getter, order=2, weight=1, mean=True, **kwargs):
        super().__init__(getter, weight, mean, **kwargs)
        self.order = order

    def penalty(self, x):
        # compute norm
        return x.norm(p=self.order)

