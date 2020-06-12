import torch
from torch import nn


def CenterLoss(feature, label, lamda):

    a = torch.tensor([[0, 0], [-1, -3], [-2, -2], [-3, 0], [-2, 3], [0, 3], [2, 3], [3, 0], [2, -2], [1, 3]])
    center = nn.Parameter(a.float(), requires_grad=True)
    # 通过标签来筛选出不同类别的中心点
    center_class = center.index_select(0, index=label.long())

    # 统计每个类别的个数
    count = torch.histc(label.float(), bins=int(max(label).item() + 1), max=int(max(label).item()))

    # 每个类别对应元素个数
    count_class = count.index_select(0, index=label.long())

    # loss = lamda/2 *((torch.mean((feature-center_class)**2).sum(dim=1)) / count_class)
    loss = lamda / 2 * (torch.mean(torch.div(torch.sum(((feature - center_class)**2), dim=1), count_class)))
    loss = lamda / 2 * (torch.mean(torch.div(torch.sum(torch.pow((feature - center_class), 2), dim=1), count_class)))
    # loss = ((feature-center_class)**2).sum(1) / center_class
    return loss
