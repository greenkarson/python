from torch import nn
import torch

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(16,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1,bias=False)
        )

    def forward(self,x):
        return self.sequential(x) + x

if __name__ == '__main__':
    net = ResNet()
    x = torch.randn(1, 16, 32, 32)
    y = net(x)
    print(y.shape)
