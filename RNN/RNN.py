import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(28 * 1, 128, 2, batch_first=True, bidirectional=False)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, h, w * c)
        h0 = torch.zeros(2 * 1, n, 128)
        c0 = torch.zeros(2 * 1, n, 128)
        hsn, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.output(hsn[:, -1, :])
        # print(hsn[:,-1,:].shape)
        return out


class MyGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(28, 128, 2, batch_first=True)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, h, w * c)
        h0 = torch.zeros(2 * 1, n, 128)
        hsn, hn = self.gru(x, h0)
        out = self.output(hsn[:, -1, :])
        return out


class GRU_Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru_cell1 = nn.GRUCell(28, 128)
        self.gru_cell2 = nn.GRUCell(128, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, h, w * c)

        hx_1 = torch.zeros(n, 128)
        hx_2 = torch.zeros(n, 128)

        for i in range(h):
            hx_1 = F.relu(self.gru_cell1(x[:, i, :], hx_1))
            hx_2 = F.relu(self.gru_cell2(hx_1, hx_2))
        out = self.output(hx_2)
        return out


if __name__ == '__main__':
    # imgs = torch.randn(2, 1, 28, 28)
    # rnn = GRU_Cell()
    # y = rnn(imgs)
    # print(y.shape)
    # exit()
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    rnn = LSTM()
    opt = optim.Adam(rnn.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100000):
        for i, (img, tag) in enumerate(train_dataloader):
            # print(img.shape)
            y = rnn(img)
            loss = loss_fn(y, tag)

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.cpu().detach().item())

        for i, (img, tag) in enumerate(test_dataset):
            y = rnn(img)
            test_loss = loss_fn(y, tag)

            print(test_loss.cpu().detach().item())
