from abc import ABC, abstractmethod

from torch import nn


class BaseRegularization(nn.Module, ABC):
    def __init__(self, getter, weight, mean):
        super().__init__()
        self.getter = getter
        self.weight = weight
        self.mean = mean
        self.step = 0

    def _get_args_and_w(self):
        if not isinstance(args := self.getter(), tuple):
            args = (args,)
        w = self.weight
        if self.mean:
            w /= len(args[0])
        return args, w

    def forward(self, loss):
        args, w = self._get_args_and_w()
        return loss + w * self.penalty(*args)

    @abstractmethod
    def penalty(self, *args):
        raise NotImplemented
