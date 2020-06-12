import torch

a = torch.tensor([0,1,0])
b = torch.tensor([0.2,0.3,0.5])

print(b[a==1])