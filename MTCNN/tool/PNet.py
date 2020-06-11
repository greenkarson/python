from torch import nn
import torch
import torch.nn.functional as F


class PNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        h = self.pre_layer(x)
        cond = F.sigmoid(self.conv4_1(h))
        offset = self.conv4_2(h)
        return cond, offset


if __name__ == '__main__':

    net = PNet()
    x = torch.randn(1, 3, 12, 12)
    cond, offset = net(x)
    print(cond.shape, offset.shape)
