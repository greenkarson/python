from torchvision import models

net = models.resnet18(pretrained=True)
print(net)