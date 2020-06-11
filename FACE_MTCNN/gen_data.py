import os
from PIL import Image
from celeba import Celeba
import numpy as np
from tools import utils


class Gendata():
    def __init__(self, metadata, output_folder, crop_size, net_stage):
        self.net_data_folder = os.path.join(output_folder, net_stage)

        self.positive_dest = os.path.join(self.net_data_folder, 'positive')
        self.negative_dest = os.path.join(self.net_data_folder, 'negative')
        self.part_dest = os.path.join(self.net_data_folder, 'part')

        [os.makedirs(x) for x in (self.positive_dest, self.negative_dest, self.part_dest) if not os.path.exists(x)]

        self.crop_size = crop_size
        self.metadata = metadata

    def run(self):
        positive_meta = open(os.path.join(self.net_data_folder, 'positive_meta.txt'), 'w')
        negative_meta = open(os.path.join(self.net_data_folder, 'negative_meta.txt'), "w")
        part_meta = open(os.path.join(self.net_data_folder, 'part_meta.txt'), 'w')

        positive_count = 0
        negative_count = 0
        part_count = 0

        for i, item in enumerate(self.metadata):

            img_path = item['file_name']
            img = Image.open(img_path)

            boxes = np.array(item['meta_data'])[:, :4]
            boxes = boxes[boxes[:, 2] >= 0]
            boxes = boxes[boxes[:, 3] >= 0]

            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            width, height = img.size
            # print(img.size,boxes.shape)

            for box in boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if max(w, h) < 40 or x1 < 0 or y1 < 0:
                    continue

                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                delta_x = np.random.randint(- w * 0.2, w * 0.2)
                delta_y = np.random.randint(- h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                crop_img = img.crop(crop_box)
                resize_img = crop_img.resize((self.crop_size, self.crop_size))
                _box = np.array([x1, y1, x2, y2])
                iou = utils.iou(_box, np.array([[nx1, ny1, nx2, ny2]]))
                if iou >= 0.65 and positive_count < 30000:
                    positive_count += 1
                    positive_meta.write(f"{positive_count}.jpg {1} {offset_x1} {offset_y1} {offset_x2} {offset_y2} \n")
                    resize_img.save(f"{self.positive_dest}/{positive_count}.jpg")
                    positive_meta.flush()

                if iou > 0.4 and part_count < 30000:
                    part_count += 1
                    part_meta.write(f"{part_count}.jpg {2} {offset_x1} {offset_y1} {offset_x2} {offset_y2} \n")
                    resize_img.save(f"{self.part_dest}/{part_count}.jpg")
                    part_meta.flush()

                size = np.random.randint(self.crop_size, min(width, height)+1 / 2)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)

                nx1 = max(x1 + delta_x, 0)
                ny1 = max(y1 + delta_y, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])
                crop_img = img.crop(crop_box)
                resize_img = crop_img.resize((self.crop_size, self.crop_size))
                _box = np.array([x1, y1, x2, y2])
                iou = utils.iou(_box, np.array([[nx1, ny1, nx2, ny2]]))
                if iou < 0.3 and negative_count < 90000:
                    negative_count += 1
                    negative_meta.write(f"{negative_count}.jpg {0} {0} {0} {0} {0} \n")
                    resize_img.save(f"{self.negative_dest}/{negative_count}.jpg")
                    negative_meta.flush()


        positive_meta.close()
        negative_meta.close()
        part_meta.close()


if __name__ == '__main__':
    celeba = Celeba(r"E:\dataset")
    train, dev, test = celeba.split_data()
    data = Gendata(test, r'F:\celeba', 12, 'pnet_eval')
    data.run()
    print(data.positive_dest)
