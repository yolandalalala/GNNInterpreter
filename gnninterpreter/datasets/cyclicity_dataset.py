import random

import networkx as nx

from gnn_xai_common.datasets import RomeDataset
from gnn_xai_common.datasets.utils import default_ax, unpack_G


class CyclicityDataset(RomeDataset):

    NODE_CLS = {
        0: 'node',
    }

    EDGE_CLS = {
        0: 'red',
        1: 'green',
    }

    GRAPH_CLS = {
        0: 'red_cyclic',
        1: 'green_cyclic',
        2: 'acyclic',
    }

    # TODO: reprocess when motif changed
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="Cyclicity", **kwargs)

    def generate(self):
        for G in super().generate():
            edge_cls = list(self.EDGE_CLS.keys())
            graph_cls = list(self.GRAPH_CLS.keys())
            neg_cls = 2
            node_cls = 0

            cls = random.choices(graph_cls, weights=[0.25, 0.25, 0.5])[0]
            if cls != neg_cls:
                pos, neg = cls, None
            else:
                pos, neg = random.sample(edge_cls, k=2)
                neg = random.choice([neg, None])
            edge = None
            while True:
                try:
                    cycle = nx.find_cycle(G)
                    edge = random.choice(cycle)
                    G.remove_edge(*edge)
                except nx.NetworkXNoCycle:
                    break
            if not edge:
                continue
            G.add_edge(*edge)
            for u, v in cycle:
                G.edges[u, v]['label'] = pos
            for u, v in G.edges:
                G.edges[u, v]['is_cycle'] = False
                if 'label' not in G.edges[u, v]:
                    G.edges[u, v]['label'] = random.choice(edge_cls)
            if cls == neg_cls:
                if neg is None:
                    G.remove_edge(*edge)
                else:
                    G.edges[edge]['label'] = neg
            else:
                for u, v in cycle:
                    G.edges[u, v]['is_cycle'] = True
            nx.set_node_attributes(G, node_cls, name='label')
            G.graph['label'] = cls
            yield G

    @default_ax
    def draw(self, G, pos=None, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw(G, pos,
                ax=ax,
                node_size=0,
                edgelist=G.edges,
                edge_color=[self.EDGE_CLS[attr['label']] for u, v, attr in G.edges(data=True)],
                node_color="k",
                width=[10 if 'is_cycle' in attr and attr['is_cycle'] else 5 for u, v, attr in G.edges(data=True)])

    def draw_gt(self, cls, ax=None):
        G = nx.generators.cycle_graph(6)
        if cls == 2:
            nx.set_edge_attributes(G, {e: random.choice([0, 1]) for e in G.edges}, "label")
        else:
            nx.set_edge_attributes(G, cls, "label")
        self.draw(G, ax=ax)

    def download(self):
        super().download()

    def process(self):
        super().process()