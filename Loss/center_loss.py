import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, class_num, feature_num):
        super(CenterLoss, self).__init__()
        self.class_num = class_num
        self.center = nn.Parameter(torch.randn(self.class_num, feature_num))

    def forward(self, feature, label):
        c = torch.index_select(self.center, 0, label)
        _n = torch.histc(label.float(), self.class_num, max=self.class_num)
        n = torch.index_select(_n, 0, label)
        d = torch.sum((feature - c) ** 2, dim=1) ** 0.5
        loss = d / n
        return loss