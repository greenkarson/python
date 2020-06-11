from torch import nn
from torchvision import models


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,7,2,padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3,2,padding=1)

        self.layer1 = self.make_layer()
        self.layer2 = self.make_layer()
        self.layer3 = self.make_layer()
        self.layer4 = self.make_layer()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,1000)
    def forward(self, x):
        pass

    def make_layer(self, block, planes, blocks, stride=1, dilate=False):
        pass


if __name__ == '__main__':
    net = models.resnet18()
    net2 = models.MobileNetV2()
    print(net2)
