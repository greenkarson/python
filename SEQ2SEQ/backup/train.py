from backup.net import *
from backup.data import *
from torch.utils.data import DataLoader
from torch import optim


class Trainer:

    def __init__(self):
        train_dataset = MyDataset("../code")
        self.train_dataloader = DataLoader(train_dataset, 100, True)

        self.net = Cnn2Seq()

        self.opt = optim.Adam(self.net.parameters())

        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self):
        for epoch in range(10):
            for i,(img,tag) in enumerate(self.train_dataloader):
                output = self.net(img)

                output = output.reshape(-1, 10)
                tag = tag.reshape(-1)

                loss = self.loss_fn(output,tag.long())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(loss)

if __name__ == '__main__':
    train = Trainer()
    train()