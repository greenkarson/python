from torch import nn
from torchvision import datasets
from torchvision import  models

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        models.MobileNetV2
