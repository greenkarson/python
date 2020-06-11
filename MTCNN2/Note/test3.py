import torch
import numpy as np

a = torch.randn(240, 137)
b = a > 0.6
c = torch.nonzero(torch.gt(a, 0.6))
print(a[a > 0.6].shape)
print(b.nonzero().shape)

o = np.array([[1, 2, 3, 4, 4], [1, 2, 3, 4, 2], [1, 2, 3, 4, 3]])
d = o[:, 4]
e = d.argsort()
print(d, e)
