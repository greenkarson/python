import torch, os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np


class MtcnnDataset(Dataset):
    def __init__(self, dataset_root, net_stage='pnet'):
        super(MtcnnDataset, self).__init__()
        self.root = dataset_root
        self.net_data_path = os.path.join(self.root, net_stage)

        self.dataset = []
        with open(f"{self.net_data_path}/positive_meta.txt", 'r') as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.net_data_path}/negative_meta.txt", 'r') as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.net_data_path}/part_meta.txt", 'r') as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.net_data_path}/landmarks_meta.txt", 'r') as f:
            self.dataset.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        strs = data.split(' ')

        if strs[1] == "0":
            img_path = f"{self.net_data_path}/negative/{strs[0]}"
            label = int(strs[1])
            box = np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
            box = torch.tensor(box).float()
            landmarks = torch.zeros(10)
        elif strs[1] == "1":
            img_path = f"{self.net_data_path}/positive/{strs[0]}"
            label = int(strs[1])
            box = np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
            box = torch.tensor(box).float()
            landmarks = torch.zeros(10)
        elif strs[1] == "2":
            img_path = f"{self.net_data_path}/part/{strs[0]}"
            label = int(strs[1])
            box = np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
            box = torch.tensor(box).float()
            landmarks = torch.zeros(10)
        else:
            img_path = f"{self.net_data_path}/landmarks/{strs[0]}"
            label = int(strs[1])
            box = torch.zeros(4).float()
            landmarks = np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5]), float(strs[6]),
                                  float(strs[7]), float(strs[8]), float(strs[9]), float(strs[10]), float(strs[11]),
                                  ])
            landmarks = torch.tensor(landmarks).float()
        img_data = self.ToTensor(Image.open(img_path))

        return img_data, label, box, landmarks

    def ToTensor(self, data):
        tf = transforms.Compose([transforms.ToTensor()])
        norm_data = (tf(data) - 127.5) * 0.0078125

        return norm_data


if __name__ == '__main__':
    dataset = MtcnnDataset(r'F:\celeba', net_stage='rnet')
    print(dataset.net_data_path)
    print(dataset[0])
