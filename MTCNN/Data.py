import torch,os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FaceDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []
        # 输入路径进行拼接，将正样本数据，负样本，部分样本数据添加到数据集
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # strip删除首位字符，split安装空格来拆分字符串
        strs = self.dataset[index].split()
        # tensor[]括号不能少，不然传入的参数类型不是张量
        cond = torch.tensor([int(strs[1])])
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])

        # 图片读取并进行归一化操作，及格式调整HWC->CHW
        if strs[1] == 1:
            img_path = f"{self.path}/positive/{strs[0]}"
        elif strs[1] == 2:
            img_path = f"{self.path}/part/{strs[0]}"
        else:
            img_path = f"{self.path}/negative/{strs[0]}"
        # 图片归一化，将归一化到-0.5左右
        img_data = torch.tensor(np.array(Image.open(img_path)) / 255.-0.5).float()
        img_data = img_data.permute(2, 0, 1)
        return img_data, cond, offset


if __name__ == '__main__':
    path = "/Users/karson/Downloads/Dataset/12"
    dataset = FaceDataset(path)
    print(dataset.__len__())
    print(dataset[0])