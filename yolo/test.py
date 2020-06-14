import torch

x = torch.randn(1,45,13,13)
y = x.permute(0,2,3,1)
reshape_y = y.reshape(y.size(0), y.size(1), y.size(2), 3, -1)
mask = reshape_y[..., 0] > 0.6
idx = mask.nonzero()
print(idx.shape)
vecs = reshape_y[mask]
print(vecs.shape)