import torch
from torch import nn


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
        )

    def forward(self, img):
        y = self.sequential(img)
        # print(y.shape)
        return y.reshape(-1)


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            # inplace 内存替换
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, nosie):
        y = self.sequential(nosie)
        return y


class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.gnet = GNet()
        self.dnet = DNet()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, noise):
        return self.gnet(noise)

    def get_D_loss(self, noise_d, real_img):
        real_y = self.dnet(real_img)
        g_img = self.gnet(noise_d)
        fake_y = self.dnet(g_img)

        real_tag = torch.ones(real_img.size(0)).cuda()
        fake_tag = torch.zeros(noise_d.size(0)).cuda()

        loss_real = self.loss_fn(real_y, real_tag)
        loss_fake = self.loss_fn(fake_y, fake_tag)

        loss_d = loss_fake + loss_real
        return loss_d

    def get_G_loss(self, noise_g):
        _g_img = self.gnet(noise_g)
        _real_y = self.dnet(_g_img)
        _real_tag = torch.ones(noise_g.size(0)).cuda()

        loss_g = self.loss_fn(_real_y, _real_tag)
        return loss_g


if __name__ == '__main__':
    net = DCGAN()
    x = torch.randn(1, 128, 1, 1)
    real = torch.randn(4, 3, 96, 96)
    loss1, loss2 = net(x, x, real)
    # print(y.shape)
