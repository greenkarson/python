import torch
from torch import nn


class Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
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


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.GRU(128 * 7 * 30, 128, 2, batch_first=True)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(-1, 128 * 7 * 30)
        x = x[:, None, :].repeat(1, 4, 1)
        h0 = torch.randn(2, x.size(0), 128)
        output, hn = self.rnn(x, h0)
        outputs = self.output_layer(output)
        return outputs


class Cnn2Seq(nn.Module):

    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self, x):
        f = self.encode(x)
        y = self.decode(f)
        return y


if __name__ == '__main__':
    # net = Encoder()
    # y = net(torch.randn(1, 3, 60, 240))
    # print(y.shape)
    # x = torch.randn([1, 32, 7, 30])
    # x = x.reshape(-1, 32 * 7 * 30)
    # x = x[:, None, :].repeat(1, 4, 1)
    # print(x.shape)

    net = Cnn2Seq()
    y = net(torch.randn(1, 3, 60, 240))
    print(y.shape)
