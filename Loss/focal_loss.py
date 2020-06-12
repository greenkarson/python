import torch
from torch import nn


class focal_loss(nn.Module):

    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x, tag):
        y = x[tag == 1]

        loss = self.alpha * (1 - y) ** self.gamma

        return torch.mean(loss)

