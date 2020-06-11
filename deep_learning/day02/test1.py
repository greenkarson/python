import torch
from torch import nn

conv = nn.Conv2d(3,16,3,1,padding=1)
x = torch.randn(1,3,16,16)
y = conv(x)
print(conv.weight)
nn.init.kaiming_normal_(conv.weight)
nn.init.normal_(0,0.1)
nn.init.zeros_(conv.bias)

print(y.shape)