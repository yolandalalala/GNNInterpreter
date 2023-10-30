from torch import nn
import torch.nn.functional as F


class ClassScoreCriterion(nn.Module):
    def __init__(self, class_idx, mode='maximize', logsoftmax=False):
        super().__init__()
        self.class_idx = class_idx
        self.mode = mode
        self.logsoftmax = logsoftmax

    def forward(self, logits):
        assert len(logits.shape) == 2
        if self.logsoftmax:
            logits = F.log_softmax(logits)
        score = logits[:, self.class_idx].mean()
        if self.mode == 'maximize':
            return -score
        elif self.mode == 'minimize':
            return score
        else:
            raise NotImplemented

