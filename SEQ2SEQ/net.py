import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 16, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0),
        )

    def forward(self, x):
        h = self.sequential(x)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128 * 7 * 30, 128, 2, batch_first=True)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(-1, 128 * 7 * 30)
        x = x[:, None, :].repeat(1, 4, 1)
        h0 = torch.zeros(2 * 1, x.size(0), 128)
        output, hn = self.rnn(x, h0)
        outputs = self.output(output)
        return outputs


class CNN2SEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        f = self.encoder(x)
        y = self.decoder(f)
        return y


if __name__ == '__main__':
    # net = Encoder()
    # net = Decoder()
    net = CNN2SEQ()
    x = torch.randn(2, 3, 60, 240)
    # x = torch.randn(2, 128, 7, 30)
    y = net(x)
    print(y.shape)