import torch

data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)

c = data[label == 1, 0]
d = data[label == 1]  #tensor([[7., 8.],[6., 5.]])
print(c)