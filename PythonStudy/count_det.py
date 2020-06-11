import numpy as np
import torch

# 计算行列式大小函数

a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))


b = torch.tensor([[1., 2.], [3., 4.]])
c = torch.tensor([[5., 6.], [7., 8.]])
m = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
k = torch.randn(1, 4, 5, 3)
print(b.det())

d = c + b
e = c - b
f = c * b
print(d)
print(e)
print(f)
print(m @ b)
print(m.t())
print(k.shape)
j = k.permute(0, 3, 1, 2)
print(j.shape)
