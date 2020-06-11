import torch

label = torch.tensor([0, 1, 0])
feature = torch.tensor([[0.6, 0.8], [0.5, 0.7], [0.3, 0.1]])
center = torch.tensor([[0.7, 0.8], [0.2, 0.3]])
# print(label.shape,feature.shape,center.shape)
c = torch.index_select(center, 0, label)
_n = torch.histc(label.float(), 2, max=2)
n = torch.index_select(_n,0,label)
print(c.shape,_n.shape)
print(c,_n,n)

d = torch.sum((feature - c)**2, dim=1)**0.5
loss = d / n
print(d)