import networkx as nx
from matplotlib import pyplot as plt


def default_ax(fn):
    def wrapper(*args, ax=None, **kwargs):
        if ax is None:
            plt.figure(figsize=[6, 6])
            plt.axis("equal")
            plt.axis("off")
            # ax = plt.axes(aspect='equal', frame_on=False)
        return fn(*args, ax=ax, **kwargs)
    return wrapper


def extract_G(super_G, idx):
    G = nx.convert_node_labels_to_integers(
        super_G.subgraph(
            n for n, attr in super_G.nodes(data=True)
            if attr['graph'] == idx
        ).copy(),
        label_attribute='id'
    )
    G.graph['label'] = super_G.graph['label'][idx]
    return G


# TODO: optimize logic
def unpack_G(super_G):
    for idx in sorted(set(nx.get_node_attributes(super_G, name='graph').values())):
        yield extract_G(super_G, idx)