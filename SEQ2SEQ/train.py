from data import *
from net import *
from torch.utils.data import DataLoader
from torch import optim, nn


class Trainer:
    def __init__(self):

        self.train_dataset = Mydataset("./code")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

        self.net = CNN2SEQ()

        self.opt = optim.Adam(self.net.parameters())

        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self):
        for epoch in range(1000):
            for i, (img, tag) in enumerate(self.train_dataloader):
                y = self.net(img)
                y = y.reshape(-1, 10)
                tag = tag.reshape(-1)
                # print(y.shape, img.shape, tag.shape)
                loss = self.loss_fn(y, tag)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(loss)


if __name__ == '__main__':
    t = Trainer()
    t()
