import os, torch
import numpy as np
from PIL import Image


class Celeba():

    def __init__(self, dataset_root):
        self.dataset_folder = os.path.join(dataset_root, "CelebA")
        self.anno_folder = os.path.join(self.dataset_folder, "Anno")
        self.image_folder = os.path.join(self.dataset_folder, "img_celeba")

        self.box_anno = os.path.join(self.anno_folder, 'list_bbox_celeba.txt')
        self.landmarks_anno = os.path.join(self.anno_folder, 'list_landmarks_celeba.txt')
        # self.file_box_anno = open(self.box_anno)
        # self.file_landmarks_anno = open(self.landmarks_anno)

    def load(self):
        file_box_anno = open(self.box_anno)
        file_landmarks_anno = open(self.landmarks_anno)
        ret = []
        for i, (file_box_line, file_landmarks_line) in enumerate(zip(file_box_anno, file_landmarks_anno)):
            if i < 2:
                continue
            image_name = file_box_line.split()[0]

            boxes = file_box_line.split()[1:]
            boxes = list(filter(lambda x: x != '', boxes))
            boxes = np.array(boxes).astype(int)

            landmarks = file_landmarks_line.split()[1:]
            landmarks = list(filter(lambda x: x != '', landmarks))
            landmarks = np.array(landmarks).astype(int)

            img_path = os.path.join(self.image_folder, image_name)
            item = {
                'file_name': img_path,
                'num_bb': 1,
                'meta_data': [boxes],
                'landmarks': [landmarks]
            }
            ret.append(item)
        return ret

    def split_data(self):
        ret = self.load()
        partition_file = os.path.join(self.dataset_folder, 'Eval', 'list_eval_partition.txt')
        f_partition = open(partition_file)

        train = []
        dev = []
        test = []

        for line, item in zip(f_partition, ret):
            dtype = int(line.split()[1])
            if dtype == 0:
                train.append(item)
            if dtype == 1:
                dev.append(item)
            if dtype == 2:
                test.append(item)
        return train, dev, test


if __name__ == '__main__':
    data = Celeba("/Users/karson/Downloads")
    train, dev, test = data.split_data()
    # print(data.dataset_folder)
    print(len(train),len(dev),len(test))
