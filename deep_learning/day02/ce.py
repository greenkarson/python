import torch
from torch import nn

class CE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,ys,tags):
        h = -ys*torch.log(tags)
        return torch.mean(h)
