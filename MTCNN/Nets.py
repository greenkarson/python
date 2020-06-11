from torch import nn
import torch


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
        # print(x.shape,x.dtype)
        h = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(h))
        offset = self.conv4_2(h)
        return cond, offset


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


class ONet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(32,64,3,1,padding=0),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,64,3,1,padding=0),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,2,1,padding=0),
            nn.PReLU()
        )
        self.fc = nn.Linear(3*3*128,256)
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