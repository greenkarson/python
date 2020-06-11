import numpy as np
import torch

a = np.array(1)
print(a.shape)
b = torch.tensor(2)
print(b.shape)
c = torch.tensor([1, 2])
print(c.shape)
d = torch.tensor([[2, 3, 4], [4, 5, 6]])
print(d.shape)
e = torch.tensor([[[3, 4, 6], [4, 5, 6]], [[7, 8, 9], [9, 10, 6]]])
print(e.shape)
