import torch,os,cv2
import numpy as np
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    # 将数据装入Dataset中，准确说将图像路径和标签装入
    def __init__(self, root, is_train=True):
        self.dataset = []  # 记录所有数据
        sub_dir = "TRAIN" if is_train else "TEST"
        # 遍历目录及子目录下标签数据及图像路径装入数据集中
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path, tag))
    # 获取数据长度，总共有多少数据
    def __len__(self):
        return len(self.dataset)
    # 处理每条数据
    def __getitem__(self, index):
        data = self.dataset[index]
        # 读入图像数据
        img_data = cv2.imread(data[0],0)
        # 图像数据展平
        img_data = img_data.reshape(-1)
        # 图像归一化
        img_data = img_data / 255

        # 标签one-hot 编码
        tag_one_hot = np.zeros(10)
        tag_one_hot[int(data[1])] = 1

        return np.float32(img_data),np.float32(tag_one_hot)


# if __name__ == '__main__':
#     dataset = MNISTDataset("../data/MNIST_IMG")
#     print(dataset[30000])

