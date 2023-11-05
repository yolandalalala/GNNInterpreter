import random
from functools import cached_property
from typing import Optional, Literal

import numpy as np
import networkx as nx
import torch
from torch import nn
from torch import distributions
import torch_geometric as pyg


class GraphSampler(nn.Module):
    """
    An i.i.d. Binomial graph sampler.
    During training, sampling employs binary Concrete distribution relaxation.
    """

    def __init__(self,
                 max_nodes: Optional[int] = None,
                 num_node_cls: Optional[int] = None,
                 num_edge_cls: Optional[int] = None,
                 nodes: Optional[list[int]] = None,
                 edges: Optional[list[tuple[int]]] = None,
                 G: nx.Graph = None,
                 learn_node_feat: bool = False,
                 learn_edge_feat: bool = False,
                 temperature: float = 1):
        """
        :param nodes: candidate nodes
        :param edges: candidate edges
        :param temperature: temperature for binary Concrete distribution relaxation
        """
        super().__init__()
        if G:
            G = nx.convert_node_labels_to_integers(G)
            nodes = [G.nodes[i]['label'] for i in range(G.number_of_nodes())]
            edges = G.edges
        self.n = max_nodes or len(nodes)
        self.k = num_node_cls or (max(nodes) if nodes is not None else 1)
        self.l = num_edge_cls
        self.nodes = nodes or self._gen_random_cls(self.n, self.k)
        self.edges = edges or self._gen_complete_edges(self.n)
        self.edge_cls = self._gen_random_cls(self.m, self.l) if num_edge_cls else None
        self.tau = temperature

        self.param_list = []

        self.omega = nn.Parameter(torch.empty(self.m))
        self.param_list.extend(["omega", "theta", "theta_pairs"])

        if learn_node_feat:
            self.xi = nn.Parameter(torch.empty(self.n, self.k))
            self.param_list.extend(["xi", "p"])
        else:
            self.xi = None

        if learn_edge_feat:
            self.eta = nn.Parameter(torch.empty(self.m, self.l))
            self.param_list.extend(["eta", "q"])
        else:
            self.eta = None

        self.init()

    @torch.no_grad()
    def init(self, G=None, eps=1e-4):
        theta = torch.rand(self.m) if G is None else torch.stack([
            torch.tensor(1 - eps if (u, v) in G.edges or (v, u) in G.edges else eps)
            for u, v in self.edge_index.T[:self.m].tolist()
        ])
        self.omega.data = torch.logit(theta)
        if self.xi is not None:
            p = distributions.Dirichlet(torch.ones(self.k)).sample([self.n]) if G is None else torch.stack([
                (torch.eye(self.k) * (1 - 2*eps) + eps)[G.nodes[i]['label']] for i in G.nodes
            ])
            self.xi.data = torch.log(p)
        if self.eta is not None:
            q = distributions.Dirichlet(torch.ones(self.l)).sample([self.m]) if G is None else torch.stack([
                (torch.eye(self.l) * (1 - 2*eps) + eps)[G.edges[(u, v)]['label']] if (u, v) in G.edges else
                (torch.eye(self.l) * (1 - 2*eps) + eps)[G.edges[(v, u)]['label']] if (v, u) in G.edges else
                torch.zeros(self.l) + eps
                for u, v in self.edge_index.T[:self.m].tolist()
            ])
            self.eta.data = torch.log(q)

    @staticmethod
    def _gen_random_cls(n, k):
        return random.choices(range(k), k=n)

    @staticmethod
    def _gen_complete_edges(n):
        return [(i, j) for i in range(n) for j in range(n) if i < j]

    @cached_property
    def m(self) -> int:
        return len(self.edges)

    @property
    def theta(self) -> torch.Tensor:
        r"""
        The latent parameter \theta for i.i.d. Binomial distributed edges.
        :return: tensor of shape [m]
        """
        return torch.sigmoid(self.omega)

    @property
    def p(self) -> torch.Tensor:
        r"""
        The latent parameter p for i.i.d. Categorical distributed nodes.
        :return: tensor of shape [n]
        """
        return torch.softmax(self.xi, dim=1)

    @property
    def q(self) -> torch.Tensor:
        r"""
        The latent parameter q for i.i.d. Categorical distributed nodes.
        :return: tensor of shape [n]
        """
        return torch.softmax(self.eta, dim=1)

    @property
    def expected_m(self) -> float:
        """
        :return: the expected number of edges in a sampled graph
        """
        return self.theta.sum().item()

    @cached_property
    def edge_index(self) -> torch.Tensor:
        """
        Generate node indices for complete edges (no self-loop) for message passing.
        :return: tensor of shape [2, 2m]
        """
        edges = ([(i, j) for i, j in self.edges] +
                 [(j, i) for i, j in self.edges])
        assert len(edges) == self.m * 2
        return torch.tensor(edges).T

    @cached_property
    def pair_index(self) -> torch.Tensor:
        """
        Generate edge indices for adjacent edge pairs.
        :return: tensor of shape [2, k]
        """
        edges = self.edge_index.T[:self.m]
        pairs = [(i, j)
                 for i in range(self.m-1)
                 for j in range(i+1, self.m)
                 if edges[i][0] == edges[j][0]]
        return torch.tensor(pairs).T

    @property
    def theta_pairs(self) -> torch.Tensor:
        return self.theta[self.pair_index]

    def to_dict(self):
        return {item: getattr(self, item) for item in self.param_list}

    def sample_eps(self, target, seed=None, expected=False):
        if expected:
            return torch.ones_like(target) / 2
        else:
            if seed is not None:
                torch.manual_seed(seed)
            else:
                torch.seed()
            return torch.rand_like(target)

    def sample_A(self, seed=None, expected=False) -> torch.Tensor:
        """
        Sample relaxed edges from binary Concrete distribution with Gumbel-Softmax trick.
        The sampled A is cached unless manually deleted.
        :return: tensor of shape [2m]
        """
        eps = self.sample_eps(self.omega, seed=seed, expected=expected)
        logistic = torch.logit(eps)
        A = torch.sigmoid((self.omega + logistic) / self.tau)
        return torch.cat([A, A], dim=0)

    def sample_X(self, seed=None, expected=False) -> torch.Tensor:
        """
        TODO
        :return:
        """
        if self.xi is not None:
            eps = self.sample_eps(self.xi, seed=seed, expected=expected)
            gumbel = -torch.log(-torch.log(eps))
            X = torch.softmax((self.xi + gumbel) / self.tau, dim=1)
            return X
        else:
            return torch.eye(self.k)[self.nodes]

    def sample_E(self, seed=None, expected=False) -> torch.Tensor:
        """
        TODO
        :return:
        """
        if self.eta is not None:
            eps = self.sample_eps(self.eta, seed=seed, expected=expected)
            gumbel = -torch.log(-torch.log(eps))
            E = torch.softmax((self.eta + gumbel) / self.tau, dim=1)
        elif self.l:
            E = torch.eye(self.l)[self.edge_cls]
        else:
            return None
        return torch.cat([E, E], dim=0)

    def forward(self, k=1,
                mode: Literal['continuous', 'discrete', 'both'] = 'continuous',
                seed=None, expected=False) -> pyg.data.Batch:
        """
        Sample a graph with i.i.d. Concrete distributed edges
        :return: a batch containing a single graph
        """
        X = self.sample_X(seed=seed, expected=expected)
        A = self.sample_A(seed=seed, expected=expected)
        E = self.sample_E(seed=seed, expected=expected)
        cont_data, disc_data = None, None
        if mode in ['continuous', 'both']:
            cont_data = pyg.data.Batch.from_data_list([pyg.data.Data(
                x=X,
                edge_index=self.edge_index,
                edge_weight=A,
                edge_attr=E,
            ) for _ in range(k)])
        if mode in ['discrete', 'both']:
            disc_data = pyg.data.Batch.from_data_list([pyg.data.Data(
                x=torch.eye(self.k)[X.argmax(dim=-1)].float(),
                edge_index=self.edge_index,
                edge_weight=(A > 0.5).float(),
                edge_attr=torch.eye(self.l)[E.argmax(dim=-1)].float() if self.eta is not None else E,
            ) for _ in range(k)])
        if mode == 'both':
            return cont_data, disc_data
        elif mode == 'continuous':
            return cont_data
        elif mode == 'discrete':
            return disc_data

    def sample_by_threshold(self, threshold) -> nx.Graph:
        r"""
        TODO
        :param threshold:
        :return:
        """

        # sample edges
        edge_list = self.edge_index.T[:self.m][self.theta >= threshold].tolist()

        # create graph
        G = nx.Graph(edge_list)
        if G.number_of_nodes() == 0:
            raise Exception("Empty graph!")
        nx.set_node_attributes(G, 0, name='label')

        # add node features
        if self.xi is not None:
            node_cls = self.xi.argmax(dim=1)
            nx.set_node_attributes(G, {v: {'label': c.item()} for v, c in enumerate(node_cls)})

        # add edge features
        if self.eta is not None:
            edge_cls = self.eta.argmax(dim=1)
            nx.set_edge_attributes(G, {(u, v): {'label': c.item()} for (u, v), c in zip(self.edges, edge_cls)})

        return G

    def _bernoulli_threshold(self):
        return torch.rand_like(self.theta)

    def _top_k_threshold(self, k):
        return self.theta.sort()[0][-k]

    def sample(self, threshold=0.5, k=None, bernoulli=False) -> pyg.data.Data:
        if k is not None:
            threshold = self._top_k_threshold(k=k)
        if bernoulli:
            threshold = self._bernoulli_threshold()
        return self.sample_by_threshold(threshold=threshold)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))