import torch
from torch import nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        h = self.sequential(x)
        h = h.view(-1, 16 * 5 * 5)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


if __name__ == '__main__':
    net = LeNet()
    a = torch.randn(2, 1, 28, 28)
    b = net(a)
    print(b.shape)
