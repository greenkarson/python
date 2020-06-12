from torch import nn
import torch

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784,32),
            nn.Linear(32,64),
            nn.Linear(64,10)

        )

    def forward(self, x):
        x = x.reshape(-1,784)
        return self.sequential(x)


if __name__ == '__main__':
    x = torch.randn(1,1,28,28)
    x.reshape(-1,784)
    net = Net()
    y = net(x)
    print(y.shape)