import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),  # 64, 31, 31
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=1),  # 64, 29, 29
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2), # 192, 29, 29
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 192, 14, 14
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 384, 14, 14
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256, 14, 14
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256, 14, 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 256, 6, 6
        )
        self.classfifier = nn.Sequential(
            # inplace 是否进行覆盖
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 200)
        )
        self.init_bias()

    def forward(self, x):
        x = self.sequential(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classfifier(x)
        return x

    def init_bias(self):

        for layer in self.sequential:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.sequential[4].bias, 1)
        nn.init.constant_(self.sequential[10].bias, 1)
        nn.init.constant_(self.sequential[12].bias, 1)


if __name__ == '__main__':
    a = torch.randn(4, 3, 64, 64)
    net = AlexNet()
    b = net(a)
    print(b.shape)
