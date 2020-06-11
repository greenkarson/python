from torch import nn
import torch


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            nn.PReLU(10),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU(32),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, stride=1)
        )
        self.box_offset = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.landmarks = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        self.apply(init_weights)

    def forward(self, x):
        feature_map = self.sequential(x)
        label = self.cls(feature_map)
        offset = self.box_offset(feature_map)
        landmarks = self.landmarks(feature_map)
        return label, offset, landmarks


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1),
            nn.PReLU(28),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(48),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(64),
        )
        self.fc = nn.Linear(64 * 3 * 3, 128)
        self.prelu = nn.PReLU(128)
        self.cls = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.box_offset = nn.Linear(128, 4)
        self.landmarks = nn.Linear(128, 10)
        self.apply(init_weights)

    def forward(self, x):
        x = self.sequential(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc(x)
        x = self.prelu(x)
        label = self.cls(x)
        offset = self.box_offset(x)
        landmarks = self.landmarks(x)
        return label, offset, landmarks


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(128),
        )
        self.fc = nn.Linear(128 * 3 * 3, 256)
        self.prelu = nn.PReLU(256)
        self.cls = nn.Sequential(
            nn.Linear(256, 2),
        )
        self.box_offset = nn.Linear(256, 4)
        self.landmarks = nn.Linear(256, 10)
        self.apply(init_weights)

    def forward(self, x):
        x = self.sequential(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc(x)
        x = self.prelu(x)
        label = self.cls(x)
        offset = self.box_offset(x)
        landmarks = self.landmarks(x)
        return label, offset, landmarks


if __name__ == '__main__':
    # net = ONet()
    # x = torch.randn(4, 3, 48, 48)
    # y = net(x)
    # print(y.shape)
    # label, offset, landmarks = net(x)
    # print(label.shape, offset.shape, landmarks.shape)
    net = RNet()
    x = torch.randn(4,3,24,24)
    label, offset, landmarks = net(x)
    print(label.shape, offset.shape, landmarks.shape)
