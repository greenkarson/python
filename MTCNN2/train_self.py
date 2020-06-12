from data import *
from Net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Train:

    def __init__(self, root, img_size):

        self.summarywrite = SummaryWriter("./runs")

        self.img_size = img_size
        # self.full_dataset = MyDataset(root, img_size)
        # train_size = int(0.8 * len(self.full_dataset))
        # test_size = len(self.full_dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, test_size])
        self.train_dataset = MyDataset(root, img_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1000, shuffle=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

        if img_size == 12:
            self.net = PNet()
        elif img_size == 24:
            self.net = RNet()
        elif img_size == 48:
            self.net = ONet()

        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, epochs):

        for epoch in range(epochs):
            total_cls_loss = 0
            total_box_loss = 0
            total_landmark_loss = 0
            for i, (img, tag) in enumerate(self.train_loader):
                predict = self.net(img)

                if self.img_size == 12:
                    predict = predict.reshape(-1, 15)
                torch.sigmoid_(predict[:, 0])

                c_mask = tag[:, 0] < 2
                c_predict = predict[c_mask]
                c_tag = tag[c_mask]
                loss_c = torch.mean((c_predict[:, 0] - c_tag[:, 0]) ** 2)

                off_mask = tag[:, 0] > 0
                off_predict = predict[off_mask]
                off_tag = tag[off_mask]
                loss_off = torch.mean((off_predict[:, 1:5] - off_tag[:, 1:5]) ** 2)

                landmark_mask = tag[:, 0] > 0
                landmark_predict = predict[landmark_mask]
                landmark_tag = tag[landmark_mask]
                loss_landmark = torch.mean((landmark_predict[:, 5:] - landmark_tag[:, 5:]) ** 2)

                loss = loss_c + loss_off + loss_landmark

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            total_cls_loss += loss_c.cpu().data.numpy()
            total_box_loss += loss_off.cpu().data.numpy()
            total_landmark_loss += loss_landmark.cpu().data.numpy()
            avr_cls_loss = total_cls_loss / i
            avr_box_loss = total_box_loss / i
            avr_landmark_loss = total_landmark_loss / i
            total_loss = avr_cls_loss + avr_box_loss + avr_landmark_loss
            self.summarywrite.add_scalars("train_loss", {"cls_loss": avr_cls_loss, "offset_loss": avr_box_loss,
                                                         "landmark_loss": avr_landmark_loss}, epoch)
            print(epoch,total_loss,avr_cls_loss,avr_box_loss,avr_landmark_loss)
            # sum_loss = loss.cpu().data.numpy() + sum_loss
            # avr_train_loss = sum_loss / len(self.train_loader)
            # print(avr_train_loss)


            '''
            sum_test_loss = 0
            for i, (img, tag) in enumerate(self.test_loader):
                predict = self.net(img)

                if self.img_size == 12:
                    predict = predict.reshape(-1, 15)

                c_mask = tag[:, 0] < 2
                c_predict = predict[c_mask]
                c_tag = tag[c_mask]
                loss_c = torch.mean((c_predict[:, 0] - c_tag[:, 0]) ** 2)

                off_mask = tag[:, 0] > 0
                off_predict = predict[off_mask]
                off_tag = tag[off_mask]
                loss_off = torch.mean((off_predict[:, 1:5] - off_tag[:, 1:5]) ** 2)

                landmark_mask = tag[:, 0] > 0
                landmark_predict = predict[landmark_mask]
                landmark_tag = tag[landmark_mask]
                loss_landmark = torch.mean((landmark_predict[:, 5:] - landmark_tag[:, 5:]) ** 2)

                test_loss = loss_c + loss_off + loss_landmark

                self.summarywrite.add_scalars("test_loss", {"test_cls_loss": loss_c, "test_offset_loss": loss_off,
                                                            "test_landmark_loss": loss_landmark}, epoch)

                # print(loss.cpu().data.numpy())
            # sum_test_loss = test_loss.cpu().data.numpy() + sum_test_loss
            # test_avr_loss = sum_test_loss / len(self.test_loader)
            # print(test_avr_loss)
            '''

            if self.img_size == 12:
                torch.save(self.net.state_dict(), "pnet.pt")
            elif self.img_size == 24:
                torch.save(self.net.state_dict(), "rnet.pt")
            elif self.img_size == 48:
                torch.save(self.net.state_dict(), "onet.pt")


if __name__ == '__main__':
    train = Train("/Users/karson/Downloads/Dataset/", 12)
    train(10000)
