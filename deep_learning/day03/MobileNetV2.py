from torch import nn
import torch

inverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class InvertResidual(nn.Module):

    def __init__(self, input_channels, output_channels, stride, expend_ratio):
        super().__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        hidden_dim = input_channels * expend_ratio
        layers = []
        if expend_ratio != 1:
            layers.extend([
                nn.Conv2d(input_channels, hidden_dim, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 1, stride, padding=0, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),

            nn.Conv2d(hidden_dim, output_channels, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        block = []
        inputc = 32
        for t, c, n, s in inverted_residual_setting:

            for i in range(n):
                stride = s if i == 0 else 1
                block.append(InvertResidual(input_channels=inputc, output_channels=c, stride=stride, expend_ratio=t))
                inputc = c

        block.extend([
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(7, 1),
            nn.Conv2d(1280, 1000, 1, 1)
        ])

        self.residual_layer = nn.Sequential(*block)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.residual_layer(h)
        # h.reshape(-1,1000)
        return h


if __name__ == '__main__':
    net = MobileNetV2()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.shape)
    print(net)
