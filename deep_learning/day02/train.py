import torch
from torchvision.transforms import Compose, Resize, RandomAffine, RandomHorizontalFlip, ToTensor, Normalize
from day02.net import *
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from PIL.Image import BICUBIC


DEVICE = "cuda:0"

class Train:

    def __init__(self):

        image_size = 224

        train_transform = Compose([
            Resize(image_size, BICUBIC),
            RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124, 117, 104)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = Compose([
            Resize(image_size, BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.summaryWriter = SummaryWriter("./logs")
        # Dataloader装载训练数据集，batch_size每轮100个数据，shuffle并打乱顺序
        self.train_dataset = datasets.CIFAR10("../data/CIFAR10/", True,transform=train_transform, download=True)
        # Dataloader装载测试数据集，batch_size每轮100个数据，shuffle并打乱顺序
        self.train_dataload = DataLoader(self.train_dataset,batch_size=100,shuffle=True)
        self.test_dataset = datasets.CIFAR10("../data/CIFAR10/",False,transform=test_transform,download=True)
        self.test_dataload = DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建网络
        self.net = NetV2()

        # 装载之前训练状态
        # self.net.load_state_dict(torch.load("./checkpoint/27.t"))
        # 将数据移动至GPU运算
        # self.net.to(DEVICE)
        # 创建优化器，将网络中net.parameters()参数放入优化器
        self.opt = optim.Adam(self.net.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self):

        global imgs
        for epoch in range(100000):
            sum_loss = 0
            for i,(imgs,tags) in enumerate(self.train_dataload):
                # imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)
                self.net.train()

                y = self.net(imgs)
                loss = self.loss_fn(y,tags)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # 将损失数据放回cpu，detach停止反向传播，item放回python标量
                sum_loss += loss.cpu().detach().item()

            avg_loss = sum_loss / len(self.train_dataload)


            sum_score = 0
            test_sum_loss = 0
            for i,(imgs,tags) in enumerate(self.test_dataload):
                # imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)
                self.net.eval()

                test_y = self.net(imgs)
                test_loss = self.loss_fn(test_y,tags)
                test_sum_loss += test_loss.cpu().detach().item()

                predict_tags = torch.argmax(test_y,dim=1)

                # 将得分数据放回cpu，detach停止反向传播，item放回python标量
                sum_score += torch.sum(torch.eq(predict_tags,tags).float()).cpu().detach().item()

            test_avg_loss = test_sum_loss / len(self.test_dataload)
            score = sum_score / len(self.test_dataset)

            # 添加训练损失与测试损失标量在tensorboard中
            self.summaryWriter.add_scalars("loss",{"train_loss":avg_loss,"test_loss":test_avg_loss},epoch)
            self.summaryWriter.add_scalar("score",score,epoch)
            self.summaryWriter.add_graph(self.net,(imgs,))
            layer1_weight = self.net.sequential[0].weight
            layer2_weight = self.net.sequential[4].weight
            layer3_weight = self.net.sequential[8].weight
            self.summaryWriter.add_histogram("layer1_weight", layer1_weight)
            self.summaryWriter.add_histogram("layer2_weight", layer2_weight)
            self.summaryWriter.add_histogram("layer3_weight", layer3_weight)

            print(epoch,avg_loss,test_avg_loss,score)
            # 保存网络训练状态
            torch.save(self.net.state_dict(),f"./checkpoint/{epoch}.t")


if __name__ == '__main__':
    train = Train()
    train()
