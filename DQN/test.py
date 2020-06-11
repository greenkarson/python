import torch
Qs = torch.tensor([[3,4],[5,6]])
a = torch.tensor([[0],[1]])

b = torch.gather(Qs,1,a)
print(Qs.max(dim=1, keepdim=True)[0])
print(b)