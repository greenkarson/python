from net import *
from data import FaceMyData
from torch.utils.data import DataLoader
from torch import optim


class Trainer:
    def __init__(self, root):
        self.dataset = FaceMyData(root)
        self.train_dataloader = DataLoader(self.dataset, batch_size=100, shuffle=True)
        self.net = DCGAN()

        self.d_opt = optim.Adam(self.net.dnet.parameters(), 0.0002, betas=(0.5, 0.9))
        self.g_opt = optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.9))

    def __call__(self):
        for epoch in range(10000):
            for i, img in enumerate(self.train_dataloader):
                real_img = img
                noise_d = torch.normal(0, 0.1, (100, 128, 1, 1))
                noise_g = torch.normal(0, 0.1, (100, 128, 1, 1))

                loss_d, loss_g = self.net(noise_d, noise_g, real_img)

                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

                print(loss_g, loss_d)


if __name__ == '__main__':
    test = Trainer("./faces")
    test()