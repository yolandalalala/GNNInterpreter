import os
import re

import networkx as nx
from tqdm.auto import tqdm
import torch_geometric as pyg

from .base_graph_dataset import BaseGraphDataset


class RomeDataset(BaseGraphDataset):
    def __init__(self, *,
                 name='Rome',
                 url='http://www.graphdrawing.org/download/rome-graphml.tgz',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        metafile = "rome/Graph.log"
        metadata_path = f'{self.raw_dir}/{metafile}'
        if os.path.exists(metadata_path):
            return list(map(lambda f: f'rome/{f}.graphml', self.get_graph_names(metadata_path)))
        else:
            return [metafile]

    @staticmethod
    def get_graph_names(logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def generate(self):
        graphmls = sorted(self.raw_paths, key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            if nx.is_connected(G):
                yield nx.convert_node_labels_to_integers(G)

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        super().process()
