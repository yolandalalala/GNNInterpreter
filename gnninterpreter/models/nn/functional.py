from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.typing import Adj


def scatter_logmeanexp(x, index, dim=-1, dim_size=None, temperature=1):
    """
    Scattered smooth maximum function
    """
    return scatter(src=x.div(temperature).exp(),
                   index=index,
                   dim=dim,
                   dim_size=dim_size,
                   reduce="mean").log().mul(temperature)


def scatter_mean_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="mean")
    return (scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum")
            / scatter(weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum"))


def scatter_sum_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="sum")
    return scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_max_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="max")
    return scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="max")


def global_mean_pool_weighted(x: Tensor,
                              batch: Optional[Tensor],
                              size: Optional[int] = None,
                              node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.mean(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).sum(dim=0, keepdim=True) / node_weight.sum())
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_mean_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def global_sum_pool_weighted(x: Tensor,
                             batch: Optional[Tensor],
                             size: Optional[int] = None,
                             node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.sum(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).sum(dim=0, keepdim=True))
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_sum_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def global_max_pool_weighted(x: Tensor,
                             batch: Optional[Tensor],
                             size: Optional[int] = None,
                             node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.max(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).max(dim=0, keepdim=True))
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_max_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def smooth_maximum_weight_propagation(edge_index: Adj,
                                      edge_weight: Tensor,
                                      size: Optional[int] = None,
                                      temperature: float = 0.05):
    size = int(edge_index.max().item() + 1) if size is None else size
    return scatter_logmeanexp(edge_weight.repeat(2),
                              index=torch.cat(tuple(edge_index)),
                              dim_size=size,
                              temperature=temperature)
