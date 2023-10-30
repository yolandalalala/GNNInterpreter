import random

import networkx as nx

from .rome_dataset import RomeDataset
from .utils import default_ax


class ColorDataset(RomeDataset):

    NODE_CLS = {
        0: 'red',
        1: 'green',
        2: 'blue',
    }

    EDGE_CLS = {
        0: 'red',
        1: 'green',
        2: 'gold',
        3: 'blue',
        4: 'magenta',
        5: 'cyan',
    }

    GRAPH_CLS = {
        0: 'negative',
        1: 'positive',
    }

    def __init__(self, seed=None, *args, **kwargs):
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        super().__init__(*args, name="Color", **kwargs)

    def _gt_edge_cls(self, G, e):
        u, v = e
        return (2**G.nodes[u]['label'] | 2**G.nodes[v]['label']) - 1

    @property
    def processed_file_names(self):
        return [f'data_color_{self.seed}.pt' if self.seed is not None else 'data_color.pt']

    def generate(self):
        for G in super().generate():
            for i in G.nodes:
                label = random.choice(list(self.NODE_CLS))
                G.nodes[i]['label'] = label
            for e in G.edges:
                G.edges[e]['label'] = self._gt_edge_cls(G, e)
            cls = random.choice(list(self.GRAPH_CLS))
            if self.GRAPH_CLS[cls] == 'negative':
                neg_size = random.randint(1, G.number_of_edges())
                for e in random.choices(list(G.edges), k=neg_size):
                    label = random.choice(list(set(self.EDGE_CLS) - {G.edges[e]['label']}))
                    G.edges[e]['label'] = label
            G.graph['label'] = cls
            yield G

    @default_ax
    def draw(self, G, pos=None, ax=None, mark_motif=True):
        pos = pos or nx.kamada_kawai_layout(G)
        true_elist = [e for e in G.edges if G.edges[e]['label'] == self._gt_edge_cls(G, e)]
        false_elist = [e for e in G.edges if G.edges[e]['label'] != self._gt_edge_cls(G, e)]
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=600,
                               node_shape='o',
                               node_color=[
                                   self.NODE_CLS[G.nodes[v]['label']]
                                   for v in G.nodes
                               ])
        nx.draw_networkx_edges(G, pos,
                               ax=ax,
                               width=16,
                               edgelist=true_elist,
                               edge_color=[
                                   self.EDGE_CLS[G.edges[e]['label']]
                                   for e in true_elist
                               ])
        nx.draw_networkx_edges(G, pos,
                               ax=ax,
                               width=16,
                               # style='dotted',
                               edgelist=false_elist,
                               edge_color=[
                                   self.EDGE_CLS[self._gt_edge_cls(G, e)]
                                   for e in false_elist
                               ])
        nx.draw_networkx_edges(G, pos,
                               ax=ax,
                               width=8,
                               # style='dotted',
                               edgelist=false_elist,
                               edge_color=[
                                   self.EDGE_CLS[G.edges[e]['label']]
                                   for e in false_elist
                               ])

    def download(self):
        super().download()

    def process(self):
        super().process()
