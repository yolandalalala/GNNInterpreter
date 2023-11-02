import torch
from torch import nn
import torch.nn.functional as F

from .nn.functional import smooth_maximum_weight_propagation, global_mean_pool_weighted, global_sum_pool_weighted
from .nn.nnconv import NNConv


class NNConvClassifier(nn.Module):
    def __init__(self, node_features, edge_features, num_classes, hidden_channels=32):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = NNConv(node_features, hidden_channels,
                            nn=nn.Linear(edge_features, node_features*hidden_channels))
        self.conv2 = NNConv(hidden_channels, hidden_channels,
                            nn=nn.Linear(edge_features, hidden_channels*hidden_channels))
        self.conv3 = NNConv(hidden_channels, hidden_channels,
                            nn=nn.Linear(edge_features, hidden_channels*hidden_channels))
        self.conv4 = NNConv(hidden_channels, hidden_channels,
                            nn=nn.Linear(edge_features, hidden_channels*hidden_channels))
        self.conv5 = NNConv(hidden_channels, hidden_channels,
                            nn=nn.Linear(edge_features, hidden_channels*hidden_channels))
        self.act = nn.LeakyReLU(inplace=True)
        self.lin1 = nn.Linear(hidden_channels*2, hidden_channels)
        self.out = nn.Linear(hidden_channels, num_classes)

    # potential bug if pass edge_weight separately
    def forward(self, batch, edge_weight=None, temperature=0.05):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        node_weight = (None if edge_weight is None
                       else smooth_maximum_weight_propagation(edge_index, edge_weight, size=len(x), temperature=temperature))

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr, edge_weight)
        x = self.act(x)
        x = self.conv2(x, edge_index, edge_attr, edge_weight)
        x = self.act(x)
        x = self.conv3(x, edge_index, edge_attr, edge_weight)
        x = self.act(x)
        x = self.conv4(x, edge_index, edge_attr, edge_weight)
        x = self.act(x)
        x = self.conv5(x, edge_index, edge_attr, edge_weight)

        # 2. Readout layer
        embeds = torch.cat([
            global_sum_pool_weighted(x, batch=batch.batch, node_weight=node_weight),
            global_mean_pool_weighted(x, batch=batch.batch, node_weight=node_weight),
        ], dim=1)

        # 3. Apply a final classifier
        x = self.lin1(embeds)
        x = self.act(x)
        x = self.out(x)

        return dict(logits=x, probs=F.softmax(x, dim=-1), embeds=embeds)
