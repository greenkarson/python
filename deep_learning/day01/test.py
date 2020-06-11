import torch
from torch.utils.data import DataLoader
from day01.data import *
from day01.net import *
tags = torch.randn(4,10)
test_y = torch.randn(4,10)
test_loss = torch.mean((tags - test_y) ** 2)
# test_loss->tensor(1.9911) tensor标量
a_argmax = torch.argmax(test_y,dim=1)
b_argmax = torch.argmax(tags,dim=1)
c = torch.eq(a_argmax,b_argmax).float()
# torch.Size([4])

h = torch.randn(4,10)
# torch.Size([4, 1])
z = torch.sum(h,dim=1,keepdim=True)
# torch.Size([4, 10])
# h/z
train_dataset = MNISTDataset("../data/MNIST_IMG",True)
train_dataloder = DataLoader(train_dataset,batch_size=100,shuffle=True)
# len(train_dataloder)
# 600
print(a_argmax.size())
print(c)
