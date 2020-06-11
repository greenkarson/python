from torch import nn
import torch
import torch.nn.functional as F


class RNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre_layrer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3, 1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1, padding=0),
            nn.PReLU()
        )
        self.fc = nn.Linear(3 * 3 * 64, 128)
        self.prelu = nn.PReLU()
        # 置信度输出
        self.detect = nn.Linear(128, 1)
        # 偏移量输出
        self.offset = nn.Linear(128, 4)

    def forward(self, x):
        h = self.pre_layrer(x)
        # h = h.reshape(-1,3*3*64)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        h = self.prelu(h)
        # 置信度输出
        label = F.sigmoid(self.detect(h))
        offset = self.offset(h)
        return label, offset


if __name__ == '__main__':
    net = RNet()
    x = torch.randn(1, 3, 24, 24)
    # y = net(x)
    # print(y.size(0))
    a, b = net(x)
    print(a.shape, b.shape)

