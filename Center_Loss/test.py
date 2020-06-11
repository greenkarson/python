import torch

data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
label = torch.tensor([0, 1, 0, 3, 4, 0], dtype=torch.float32)

c = torch.index_select(data,0,label.long())
count = torch.histc(label,bins=5,max=4)
count_class = count.index_select(dim=0, index=label.long())
print(c)
print(count)
print(count_class)