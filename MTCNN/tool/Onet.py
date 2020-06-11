import torch
import torch.nn.functional as F
from torch import nn


class ONet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 1, 3, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 2, 1, padding=0),
            nn.PReLU()
        )
        self.fc = nn.Linear(3 * 3 * 128, 256)
        self.prelu = nn.PReLU()
        # 置信度输出
        self.detect = nn.Linear(256, 1)
        # 偏移量输出
        self.offset = nn.Linear(256, 4)

    def forward(self, x):
        h = self.pre_layer(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        h = self.prelu(h)
        label = F.sigmoid(self.detect(h))
        offset = self.offset(h)
        return label, offset


if __name__ == '__main__':
    net = ONet()
    x = torch.randn(1, 3, 48, 48)
    # y = net(x)
    # print(y.shape)
    a, b = net(x)
    print(a.shape, b.shape)
