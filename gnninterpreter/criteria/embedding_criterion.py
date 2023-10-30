import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingCriterion(nn.Module):
    def __init__(self, target_embedding):
        super().__init__()
        self.target = target_embedding

    def forward(self, embeds):
        assert len(embeds.shape) == 2
        return (1 - F.cosine_similarity(self.target[None, :], embeds)).mean()
