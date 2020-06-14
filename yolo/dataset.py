import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from torchvision import transforms
import os
import math

LABEL_FILE_PATH = 'data/person_label.txt'
IMG_BASE_DIR = 'data'

tf = transforms.Compose([
    transforms.ToTensor()
])
def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1
    return b
class MyDataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()
    
    def __len__(self):
        return len(self.dataset)
    
    
class Net():