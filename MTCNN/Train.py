import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from Data import FaceDataset
import Nets


class Trainer:
    def __init__(self, net, dataset_path):
        self.net = net
        self.dataset_path = dataset_path

        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()

        self.optim = optim.Adam(self.net.parameters())

    def __call__(self):
        dataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2, drop_last=True)

        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                _output_category, _output_offset = self.net(img_data_)

                # print(category_.shape, offset_.shape)
                # print(_output_category.shape, _output_offset.shape)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)

                # print(output_category.shape, output_offset.shape)
                # print("----------------------------------------")

                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask).float()
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss(output_category, category)

                offset_mask = torch.gt(category_, 0)
                offset_index = torch.nonzero(offset_mask)[:, 0]
                offset = offset_[offset_index]
                output_offset = output_offset[offset_index]
                offset_loss = self.offset_loss(output_offset, offset)

                # 总损失
                loss = cls_loss + offset_loss

                print("i=", i, "loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(),
                      " offset_loss",
                      offset_loss.cpu().data.numpy())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


if __name__ == '__main__':
    net = Nets.PNet()
    train = Trainer(net, "/Users/karson/Downloads/Dataset/12")
    train()
