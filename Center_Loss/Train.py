import torch
from torch.utils.data import DataLoader
import torchvision.datasets as Dataset
import torchvision.transforms as transform
import torch.nn as nn
import os
from Model import CenterLossNet
import Centerloss
import torch.optim as optimizer
import Drawing as Draw


class Train:
    def __init__(self):
        self.save_path = "Models/_weight.pth"
        self.batch_size = 1000
        self.net = CenterLossNet().cuda() if torch.cuda.is_available else CenterLossNet().cpu()
        self.transform = transform.Compose([transform.ToTensor(), transform.Normalize(mean=[0.5], std=[0.5])])
        self.dataset = Dataset.MNIST("./MNIST", train=True, download=True, transform=self.transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        self.classification_loss = nn.NLLLoss()
        self.centerloss = Centerloss
        self.optimizer = optimizer.SGD(self.net.parameters(), lr=2e-3, momentum=0.9)

    def trainer(self):
        if os.path.exists(self.save_path):
            print("Parameters Exists")
            self.net.load_state_dict(torch.load(self.save_path))

        # epoch = 100
        for epoch in range(100000):
            Features = []
            Labels = []
            for i, (dataset, target) in enumerate(self.dataloader):
                dataset, target = dataset.cuda(), target.cuda()
                features, output = self.net(dataset)

                Features.append(features)
                Labels.append(target)

                center_loss = self.centerloss.centerloss(features, target, 2)
                class_loss = self.classification_loss(output, target)
                loss = center_loss + class_loss

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                # print(self.net.features.bias.grad)


            print("epoch:{}---loss:{}---center_loss:{}---class_loss:{}".format(epoch, loss, center_loss, class_loss))
            feature_list = torch.cat(Features, dim=0)
            label_list = torch.cat(Labels, dim=0)

            # print(feature_list.shape)
            # print(label_list.shape)
            Draw.DrawPics(feature_list.data.cpu().numpy(), label_list.data.cpu().numpy(), epoch)

            torch.save(self.net.state_dict(), self.save_path)


if __name__ == "__main__":
    train = Train()
    train.trainer()
    # print(train.dataset.__len__())