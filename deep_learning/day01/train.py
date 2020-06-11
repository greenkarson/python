import torch
from day01.net import *
from day01.data import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda:0"


class Train:

    def __init__(self, root):

        self.summaryWriter = SummaryWriter("./logs")
        # Dataloader装载训练数据集，batch_size每轮100个数据，shuffle并打乱顺序
        self.train_dataset = MNISTDataset(root, True)
        self.train_dataload = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
        # Dataloader装载测试数据集，batch_size每轮100个数据，shuffle并打乱顺序
        self.test_dataset = MNISTDataset(root, False)
        self.test_dataload = DataLoader(self.test_dataset, batch_size=100, shuffle=True)
        # 创建网络
        self.net = NetV1()

        # 装载之前训练状态
        # self.net.load_state_dict(torch.load("./checkpoint/27.t"))
        # 将数据移动至GPU运算
        # self.net.to(DEVICE)
        # 创建优化器，将网络中net.parameters()参数放入优化器
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):

        for epoch in range(100000):
            sum_loss = 0
            for i, (imgs, tags) in enumerate(self.train_dataload):
                # imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)
                self.net.train()

                y = self.net(imgs)
                loss = torch.mean((tags - y) ** 2)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # 将损失数据放回cpu，detach停止反向传播，item放回python标量
                sum_loss += loss.cpu().detach().item()

            avg_loss = sum_loss / len(self.train_dataload)

            sum_score = 0
            test_sum_loss = 0
            for i, (imgs, tags) in enumerate(self.test_dataload):
                # imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)
                self.net.eval()

                test_y = self.net(imgs)
                test_loss = torch.mean((tags - test_y) ** 2)
                test_sum_loss += test_loss.cpu().detach().item()

                predict_tags = torch.argmax(test_y, dim=1)
                label_tags = torch.argmax(tags, dim=1)
                # 将得分数据放回cpu，detach停止反向传播，item放回python标量
                sum_score += torch.sum(torch.eq(predict_tags, label_tags).float()).cpu().detach().item()

            test_avg_loss = test_sum_loss / len(self.test_dataload)
            score = sum_score / len(self.test_dataset)

            self.summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
            self.summaryWriter.add_scalar("score", score, epoch)
            self.summaryWriter.add_graph(self.net, (imgs,))

            print(epoch, avg_loss, test_avg_loss, score)
            # 保存网络训练状态
            torch.save(self.net.state_dict(), f"./checkpoint/{epoch}.t")


if __name__ == '__main__':
    train = Train("../data/MNIST_IMG")
    train()
