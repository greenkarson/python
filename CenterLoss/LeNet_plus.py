import torch
from torch import nn


class LeNet_Plus(nn.Module):
    def __init__(self):
        super(LeNet_Plus, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.feature = nn.Linear(128*3*3,2)
        self.output = nn.Linear(2,10)

    def forward(self, x):
        x = self.sequential(x)
        x = x.view(-1,128*3*3)
        features = self.feature(x)
        outputs = self.output(features)
        return features, outputs


if __name__ == '__main__':
    net = LeNet_Plus()
    a = torch.randn(2,1,28,28)
    b, c = net(a)
    print(b.shape, c.shape)