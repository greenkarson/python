from torch.utils.data import Dataset
import cv2, os, torch
import numpy as np


class FaceMyData(Dataset):
    def __init__(self, root):
        self.root = root
        self.dataset = os.listdir(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filename = self.dataset[index]
        img_data = cv2.imread(f"{self.root}/{filename}")
        img_data = img_data[..., ::-1]
        img_data = img_data.transpose(2, 0, 1)
        # 生成网络需要-1，1直径数据调整输出
        img_data = ((img_data / 255. -0.5)*2).astype(np.float32)
        # print(img_data.shape)
        return img_data


if __name__ == '__main__':
    dataset = FaceMyData("./faces")
    print(dataset[0])
