import numpy as np
import torch

a = np.diag([1,2,3,4])
print(a)
b = torch.diag(torch.tensor([1,2,3,4]))
print(b)
c = np.eye(4,3)
print(c)
d = torch.eye(3,4)
print(d)
e = np.tri(3,3)
print(e)
f = torch.tril(torch.ones(3,3))
print(f)
g = np.ones([3,3])
print(g)
h = torch.zeros(3,3)
print(h)