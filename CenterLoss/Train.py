import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import nn
from LeNet_plus import LeNet_Plus
from Centerloss import CenterLoss
from torch import optim
from Drawing import DrawPics


class Train():
    def __init__(self):

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.dataset = datasets.MNIST("./MNIST",train=True,download=True, transform=self.transform)
        self.Dataloader = DataLoader(self.dataset,batch_size=100,shuffle=True,num_workers=4)

        self.net = LeNet_Plus()

        self.classifiction_loss = nn.CrossEntropyLoss()
        self.centerloss = CenterLoss

        self.opt = optim.SGD(self.net.parameters(), lr=2e-3)

    def __call__(self, *args, **kwargs):

        for epoch in range(10000):
            Features = []
            Lables = []
            for i, (img, tag) in enumerate(self.Dataloader):

                features, output = self.net(img)

                Features.append(features)
                Lables.append(tag)

                # print(features.shape,tag.shape)

                center_loss = self.centerloss(features, tag, 2)
                class_loss = self.classifiction_loss(output, tag)
                loss = center_loss + class_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print(f"epoch:{epoch}---loss:{loss}---center_loss:{center_loss}---class_loss{class_loss}")
            features_list = torch.cat(Features, dim=0)
            label_list = torch.cat(Lables, dim=0)
            DrawPics(features_list.cpu().detach().numpy(),label_list.cpu().detach().numpy(),epoch)


if __name__ == '__main__':
    train = Train()
    train()
