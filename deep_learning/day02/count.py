import torch,thop
from torch import nn

conv = nn.Conv2d(3,16,3,1,padding=1)
x = torch.randn(1,3,16,16)
y = conv(x)
# 统计计算量
flops, parm = thop.profile(conv,(x,))
# 输出格式
flops, parm = thop.clever_format((flops,parm),'%3.f')
print(flops,parm)