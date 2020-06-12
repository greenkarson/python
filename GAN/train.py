from net import *
from data import FaceMyData
from torch.utils.data import DataLoader
from torch import optim
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
import time


class Trainer:
    def __init__(self, root):
        self.summarywrite = SummaryWriter("./runs")
        self.dataset = FaceMyData(root)
        self.train_dataloader = DataLoader(self.dataset, batch_size=1000, shuffle=True, num_workers=4)
        self.net = DCGAN().cuda()
        self.net.load_state_dict(torch.load("./param/param2020-05-15-19-53-49.pt"))

        self.d_opt = optim.Adam(self.net.dnet.parameters(), 0.0002, betas=(0.5, 0.9))
        self.g_opt = optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.9))

    def __call__(self):
        for epoch in range(10000):

            for i, img in enumerate(self.train_dataloader):
                real_img = img.cuda()
                noise_d = torch.normal(0, 1, (100, 128, 1, 1)).cuda()
                loss_d = self.net.get_D_loss(noise_d, real_img)

                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                noise_g = torch.normal(0, 1, (100, 128, 1, 1)).cuda()
                loss_g = self.net.get_G_loss(noise_g)

                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

                print(i, loss_d.cpu().detach().item(), loss_g.cpu().detach().item())

            print("........................")
            noise = torch.normal(0, 1, (8, 128, 1, 1)).cuda()
            y = self.net(noise)
            utils.save_image(y, f"./gen_img/{epoch}.jpg", range=(-1, 1), normalize=True)

            self.summarywrite.add_scalars("loss", {"d_loss": loss_d.cpu().detach().item(),
                                                   "g_loss": loss_g.cpu().detach().item()},
                                          epoch)
            # self.summarywrite.add_graph(self.net,(real_img,))
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            torch.save(self.net.state_dict(), f"./param/param{t}.pt")


if __name__ == '__main__':
    test = Trainer("./faces")
    test()
