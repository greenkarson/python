import torch
from torch import nn


class NetV1(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,padding=1)
        )
        self.outlayer = nn.Sequential(
            nn.Linear(64*8*8,10)
        )

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1,64*8*8)
        h = self.outlayer(h)
        return h


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NetV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.outlayer = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(64*8*8,10)
        )
        self.apply(weight_init)

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1,64*8*8)
        h = self.outlayer(h)
        return h

class NetV3(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(256*4*4,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1,256*4*4)
        h = self.out_layer(h)
        return h

class NetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(3,2,padding=1),
            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.out_layer = nn.Sequential(
            nn.Linear(256*4*4,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1,256*4*4)
        h = self.out_layer(h)
        return h


if __name__ == '__main__':
    net = NetV4()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
