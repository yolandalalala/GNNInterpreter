import random

import networkx as nx

from .rome_dataset import RomeDataset
from .utils import default_ax


class MotifDataset(RomeDataset):

    NODE_CLS = {
        0: 'red',
        1: 'orange',
        2: 'green',
        3: 'blue',
        4: 'magenta',
    }

    GRAPH_CLS = {
        0: 'partial',
        1: 'house',
        2: 'house_x',
        3: 'comp_4',
        4: 'comp_5',
    }

    class MotifFactory:

        @staticmethod
        def partial():
            motif = random.choice([
                MotifDataset.MotifFactory.house,
                MotifDataset.MotifFactory.house_x,
                MotifDataset.MotifFactory.comp_4,
                MotifDataset.MotifFactory.comp_5,
            ])()
            if random.random() < 1:
                # remove random edge
                motif.remove_edge(*random.choice(list(motif.edges)))
            else:
                # replace node label
                node = random.choice(list(motif.nodes))
                original = motif.nodes[node]['label']
                candidates = set(MotifDataset.NODE_CLS.keys()) - {original}
                motif.nodes[node]['label'] = random.choice(list(candidates))
            return motif

        @staticmethod
        def house():
            motif = nx.house_graph()
            nx.set_node_attributes(motif, {
                i: {'label': i} for i in motif.nodes()
            })
            return motif

        @staticmethod
        def house_x():
            motif = nx.house_x_graph()
            nx.set_node_attributes(motif, {
                i: {'label': i} for i in motif.nodes()
            })
            return motif

        @staticmethod
        def comp_4():
            motif = nx.complete_graph(4)
            nx.set_node_attributes(motif, {
                i: {'label': i} for i in motif.nodes()
            })
            return motif

        @staticmethod
        def comp_5():
            motif = nx.complete_graph(5)
            nx.set_node_attributes(motif, {
                i: {'label': i} for i in motif.nodes()
            })
            return motif

    def get_motif(self, cls):
        motif = getattr(self.MotifFactory, self.GRAPH_CLS[cls])()
        nx.set_node_attributes(motif, name='is_motif', values=True)
        motif.graph['label'] = cls
        return self.convert(motif)

    # TODO: reprocess when motif changed
    def  __init__(self, name="Motif", **kwargs):
        super().__init__(name=name, **kwargs)

    def generate(self):
        for G in super().generate():
            nx.set_node_attributes(G, name='label',
                                   values={v: random.choice(list(self.NODE_CLS)) for v in G.nodes})
            nx.set_node_attributes(G, name='is_motif', values=False)
            cls = random.choice(list(self.GRAPH_CLS.keys()))
            if (motif := self.get_motif(cls).G).number_of_nodes() > 0:
                G = nx.disjoint_union(motif, G)
                boundary = motif.number_of_nodes()
                motif_node = random.choice(list(G.nodes)[:boundary])
                graph_node = random.choice(list(G.nodes)[boundary:])
                G.add_edge(motif_node, graph_node)
            G.graph['label'] = cls
            yield G

    @default_ax
    def draw(self, G, pos=None, ax=None, mark_motif=True):
        pos = pos or nx.kamada_kawai_layout(G)
        node_attr = nx.get_node_attributes(G, 'is_motif')
        motif_nodes = set(filter(node_attr.__getitem__, node_attr))
        reg_nodes = set(G.nodes) - motif_nodes
        for nodes, shape in [(motif_nodes, 'D' if mark_motif else 'o'),
                             (reg_nodes, 'o')]:
            nx.draw_networkx_nodes(G, pos,
                                   ax=ax,
                                   nodelist=nodes,
                                   node_size=800,
                                   node_shape=shape,
                                   node_color=[
                                       self.NODE_CLS[G.nodes[v]['label']]
                                       for v in nodes
                                   ])
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)
        nx.draw_networkx_edges(G.subgraph(motif_nodes), pos, ax=ax, width=(12 if mark_motif else 6))

    def draw_gt(self, cls, ax=None):
        self.draw(self.get_motif(cls).G, ax=ax, mark_motif=False)

    def download(self):
        super().download()

    def process(self):
        super().process()