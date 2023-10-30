import random

import networkx as nx
import numpy as np
from tqdm.auto import trange

from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax


class ShapeDataset(BaseGraphDataset):

    NODE_CLS = {
        0: 'node',
    }

    GRAPH_CLS = {
        0: 'random',
        1: 'lollipop',
        2: 'wheel',
        3: 'grid',
        4: 'star',
    }

    class GraphFactory:

        @staticmethod
        def random(n_min=8, n_max=32, p_min=0.2, p_max=1):
            n = random.randint(n_min, n_max)
            p = random.uniform(p_min, p_max)
            return nx.generators.gnp_random_graph(n, p)

        @staticmethod
        def lollipop(m_min=4, m_max=16, n_min=4, n_max=16):
            m = random.randint(m_min, m_max)
            n = random.randint(n_min, n_max)
            return nx.generators.lollipop_graph(m, n)

        @staticmethod
        def wheel(n_min=4, n_max=64):
            n = random.randint(n_min, n_max)
            return nx.generators.wheel_graph(n)

        @staticmethod
        def ladder(n_min=4, n_max=32):
            n = random.randint(n_min, n_max)
            return nx.generators.circular_ladder_graph(n)

        @staticmethod
        def star(n_min=4, n_max=64):
            n = random.randint(n_min, n_max)
            return nx.generators.star_graph(n)

        @staticmethod
        def cycle(n_min=4, n_max=64):
            n = random.randint(n_min, n_max)
            return nx.generators.cycle_graph(n)

        @staticmethod
        def grid(x_min=2, x_max=8, y_min=2, y_max=8):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            return nx.generators.grid_graph((x, y))

        @staticmethod
        def tree(n_min=4, n_max=64):
            n = random.randint(n_min, n_max)
            return nx.generators.trees.random_tree(n=n, seed=random.randrange(1000000))

        @staticmethod
        def path(n_min=2, n_max=64):
            n = random.randint(n_min, n_max)
            return nx.generators.classic.path_graph(n)

        @staticmethod
        def rary_tree(r_min=2, r_max=8, n_min=2, n_max=64):
            r = random.randint(r_min, r_max)
            n = random.randint(n_min, n_max)
            return nx.generators.classic.full_rary_tree(r, n)

    def __init__(self, *,
                 name=f'Topology',
                 size=8000,
                 **kwargs):
        self.size = size
        super().__init__(name=name, **kwargs)

    def add_irregular_edges(self, G):
        new_edge_perc = np.random.uniform(high=0.2)
        total_edges = G.number_of_edges()
        for i in range(int(np.round(new_edge_perc * total_edges))):
            #randomly select the starting node
            start = np.random.choice(np.array(G.nodes))
            prob = np.random.uniform()
            if prob >= 0.5:
                #select ending nodes from exisiting node set
                end = np.random.choice(np.delete(G.nodes, start))
            else:
                #create new nodes
                end = G.number_of_nodes()
            G.add_edge(start,end)

    def generate(self):
        for _ in trange(self.size):
            node_cls = 0
            graph_cls = random.choice(list(self.GRAPH_CLS.keys()))
            G = getattr(self.GraphFactory, self.GRAPH_CLS[graph_cls])()
            G = nx.convert_node_labels_to_integers(G)
            self.add_irregular_edges(G)
            nx.set_node_attributes(G, node_cls, name='label')
            G.graph['label'] = graph_cls
            yield G

    @default_ax
    def draw(self, G, pos=None, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw(G, pos,
                ax=ax,
                node_size=50,
                node_color="k",
                width=2,
                edgelist=G.edges)

    def draw_gt(self, cls, *args, ax=None, **kwargs):
        self.draw(getattr(self.GraphFactory, self.GRAPH_CLS[cls])(*args, **kwargs), ax=ax)

    def process(self):
        super().process()
