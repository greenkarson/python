from data import *
from Net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time


class Train:

    def __init__(self, root, img_size):
        self.summarywrite = SummaryWriter("./runs")

        self.img_size = img_size

        self.mydataset = MyDataset(root, img_size)
        self.dataloader = DataLoader(self.mydataset, batch_size=1000, shuffle=True, num_workers=4)

        if img_size == 12:
            self.net = PNet().cuda()
            self.net.load_state_dict(torch.load(r"param\pnet2020-05-15-21-43-13.pt"))
        elif img_size == 24:
            self.net = RNet().cuda()
            self.net.load_state_dict(torch.load(r"param\rnet2020-05-16-11-26-57.pt"))
        elif img_size == 48:
            self.net = ONet().cuda()
            self.net.load_state_dict(torch.load(r"param\onet2020-05-15-21-52-22.pt"))

        self.opt = optim.Adam(self.net.parameters(), lr=1e-4)

    def __call__(self, epochs):

        for epoch in range(epochs):
            total_loss = 0
            for i, (img, tag) in enumerate(self.dataloader):
                img = img.cuda()
                tag = tag.cuda()

                predict = self.net(img)

                if self.img_size == 12:
                    predict = predict.reshape(-1, 15)

                torch.sigmoid_(predict[:, 0])

                c_mask = tag[:, 0] < 2
                # print(c_mask.shape)

                c_predict = predict[c_mask]
                c_tag = tag[c_mask]
                loss_c = torch.mean((c_predict[:, 0] - c_tag[:, 0]) ** 2)

                off_mask = tag[:, 0] > 0
                off_predict = predict[off_mask]
                off_tag = tag[off_mask]
                loss_off = torch.mean((off_predict[:, 1:5] - off_tag[:, 1:5]) ** 2)

                loss = loss_c + loss_off

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(i, loss.cpu().data.numpy())
            total_loss += loss.cpu().detach().data
            avr_loss = total_loss / i
            self.summarywrite.add_scalars("train_loss", {"total_avr_loss": avr_loss}, epoch)

            if self.img_size == 12:
                t_pnet = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                torch.save(self.net.state_dict(), f"./param/pnet{t_pnet}.pt")
            elif self.img_size == 24:
                t_rnet = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                torch.save(self.net.state_dict(), f"./param/rnet{t_rnet}.pt")
            elif self.img_size == 48:
                t_onet = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                torch.save(self.net.state_dict(), f"./param/onet{t_onet}.pt")


if __name__ == '__main__':
    train = Train("D:\work\Dataset", 24)
    train(100000)
