import torch
from torch import nn
import torch_geometric as pyg
import torch.nn.functional as F

from .nn.functional import smooth_maximum_weight_propagation, global_sum_pool_weighted, global_mean_pool_weighted


class GNNClassifier(nn.Module):
    def __init__(self, hidden_channels, node_features, num_classes, model=pyg.nn.GCN, num_layers=3, dropout=0):
        super().__init__()
        self.conv = model(in_channels=node_features,
                          hidden_channels=hidden_channels,
                          num_layers=num_layers,
                          act=nn.LeakyReLU(inplace=True),
                          dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.lin = pyg.nn.Linear(hidden_channels*2, hidden_channels)
        self.out = pyg.nn.Linear(hidden_channels, num_classes)

    def forward(self, batch=None, embeds=None, feats=None, edge_weight=None, temperature=0.05):
        if embeds is None and feats is None:
            node_weight = (None if edge_weight is None
                           else smooth_maximum_weight_propagation(batch.edge_index, edge_weight,
                                                                  size=len(batch.x),
                                                                  temperature=temperature))

            # 1. Obtain node embeddings
            h = self.conv(batch.x, batch.edge_index, edge_weight=edge_weight)

            # 2. Readout layer
            feats = torch.cat([
                global_sum_pool_weighted(h, batch=batch.batch, node_weight=node_weight),
                global_mean_pool_weighted(h, batch=batch.batch, node_weight=node_weight),
            ], dim=1)

        if embeds is None:
            # 3. Apply a final classifier
            h = self.drop(feats)
            embeds = self.lin(h)

        hout = h = embeds.relu()
        h = self.out(h)

        return dict(logits=h, probs=F.softmax(h, dim=-1), feats=feats, embeds=embeds, hout=hout)