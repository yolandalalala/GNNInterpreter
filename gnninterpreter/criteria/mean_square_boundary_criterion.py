from torch import nn


class MeanSquareBoundaryCriterion(nn.Module):
    def __init__(self, class_a, class_b):
        super().__init__()
        self.class_a = class_a
        self.class_b = class_b

    def forward(self, logits):
        return (logits[:, self.class_a] - logits[:, self.class_b]).square().mean()

