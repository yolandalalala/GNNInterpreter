from typing import Optional

import torch
from torch import Tensor

from .functional import scatter_mean_weighted


class GraphNormWeighted(torch.nn.Module):

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, node_weight: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean_weighted(x, batch, dim=0, dim_size=batch_size, weight=node_weight)
        out = x - mean.index_select(0, batch) * self.mean_scale
        var = scatter_mean_weighted(out.pow(2), batch, dim=0, dim_size=batch_size, weight=node_weight)
        std = (var + self.eps).sqrt().index_select(0, batch)
        return self.weight * out / std + self.bias

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'