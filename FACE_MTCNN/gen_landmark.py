import os
from PIL import Image
from celeba import Celeba
import numpy as np
from tools import utils


class Genlandmark():
    def __init__(self, metadata, output_folder, crop_size, net_stage):
        self.net_data_folder = os.path.join(output_folder, net_stage)

        self.landmarks_dest = os.path.join(self.net_data_folder, 'landmarks')
        if not os.path.exists(self.landmarks_dest):
            os.makedirs( self.landmarks_dest)

        self.crop_size = crop_size
        self.metadata = metadata

    def run(self):
        landmarks_meta = open(os.path.join(self.net_data_folder, 'landmarks_meta.txt'), 'w')

        landmarks_count = 0

        for i, item in enumerate(self.metadata):

            img_path = item['file_name']
            boxes = item['meta_data']
            landmarks = item['landmarks']

            img = Image.open(img_path)
            width, height = img.size

            for bbox, landmark in zip(boxes, landmarks):
                left = bbox[0]
                top = bbox[1]
                w = bbox[2]
                h = bbox[3]

                # there is error data in datasets
                if w <= 0 or h <= 0:
                    continue

                right = bbox[0] + w + 1
                bottom = bbox[1] + h + 1

                crop_box = np.array([left, top, right, bottom])
                crop_img = img.crop(crop_box)
                resize_img = crop_img.resize((self.crop_size, self.crop_size))

                landmark = np.array(landmark)
                landmark.resize(5, 2)

                # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
                landmark_gtx = (landmark[:, 0] - left) / w
                landmark_gty = (landmark[:, 1] - top) / h
                landmark_gt = np.concatenate([landmark_gtx, landmark_gty]).tolist()
                if landmarks_count < 60000:
                    landmarks_count += 1
                    resize_img.save(f"{self.landmarks_dest}/{landmarks_count}.jpg")
                    landmarks_meta.write(f"{landmarks_count}.jpg {3} ")
                    landmarks_meta.write(" ".join([str(i) for i in landmark_gt]))
                    landmarks_meta.write('\n')
                    landmarks_meta.flush()

        landmarks_meta.close()


if __name__ == '__main__':
    celeba = Celeba(r"E:\dataset")
    train, dev, test = celeba.split_data()
    data = Genlandmark(train, r'F:\celeba', 48, 'onet')
    data.run()

