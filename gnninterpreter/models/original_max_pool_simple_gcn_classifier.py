from torch import nn
import torch_geometric as pyg
import torch.nn.functional as F

class OriginalMaxPoolSimpleGCNClassifier(nn.Module):
    def __init__(self, hidden_channels, node_features, num_classes, num_layers, norm=None, act=nn.LeakyReLU, dropout=0):
        super().__init__()
        self.conv = pyg.nn.GCN(in_channels=node_features,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               act=act(),
                               norm=(norm(in_channels=hidden_channels)
                                     if norm is not None else None),
                               dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.lin = pyg.nn.Linear(hidden_channels, hidden_channels)
        self.act = act()
        self.out = pyg.nn.Linear(hidden_channels, num_classes)

    def forward(self, batch, edge_weight=None, temperature=0.05):

        # 1. Obtain node embeddings
        h = self.conv(batch.x, batch.edge_index, edge_weight)

        # 2. Readout layer
        embeds = h = pyg.nn.global_max_pool(h, batch=batch.batch)

        # 3. Apply a final classifier
        h = self.drop(h)
        h = self.lin(h)
        h = self.act(h)
        h = self.out(h)

        return dict(logits=h, probs=F.softmax(h, dim=-1), embeds=embeds)
