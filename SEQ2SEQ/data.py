import torch,os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
tf = transforms.ToTensor()


class Mydataset(Dataset):
    def __init__(self, root):
        self.dataset = os.listdir(root)
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filename = self.dataset[index]
        strs = filename.split(".")[0]
        label = np.array([int(x) for x in strs])
        img = tf(Image.open(f"{self.root}/{filename}"))
        return img, label


if __name__ == '__main__':
    dataset = Mydataset("./code")
    print(dataset[0])