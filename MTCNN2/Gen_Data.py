import os
from PIL import Image
import numpy as np
from tools import utils


class GenData():

    def __init__(self, src_root, gen_root, image_size):
        self.image_size = image_size

        self.src_image_path = f"{src_root}/img_celeba.7z"
        self.src_anno_path = f"{src_root}/Anno/list_bbox_celeba.txt"
        self.src_landmark_path = f"{src_root}/Anno/list_landmarks_celeba.txt"

        self.positive_image_dir = f"{gen_root}/{image_size}/positive"
        self.negative_image_dir = f"{gen_root}/{image_size}/negative"
        self.part_image_dir = f"{gen_root}/{image_size}/part"

        self.positive_label = f"{gen_root}/{image_size}/positive.txt"
        self.negative_label = f"{gen_root}/{image_size}/negative.txt"
        self.part_label = f"{gen_root}/{image_size}/part.txt"

        # 若文件夹不存在则创建路径
        for path in [self.positive_image_dir, self.negative_image_dir, self.part_image_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def run(self, epoch):

        positive_label_txt = open(self.positive_label, "w")
        negative_label_txt = open(self.negative_label, "w")
        part_label_txt = open(self.part_label, "w")

        positive_count = 0
        negative_count = 0
        part_count = 0

        for _ in range(epoch):
            box_nano = open(self.src_anno_path, "r")
            landmark_anno = open(self.src_landmark_path, "r")
            # for i, line in enumerate(open(self.src_anno_path, "r")):
            for i, (box_line, landmark_line) in enumerate(zip(box_nano, landmark_anno)):
                if i < 2:
                    continue
                strs = box_line.split()
                landmarks = landmark_line.split()

                print(strs, landmarks)

                img_path = f"{self.src_image_path}/{strs[0]}"
                img = Image.open(img_path)
                # img.show()

                x1 = int(strs[1])
                y1 = int(strs[2])
                w = int(strs[3])
                h = int(strs[4])
                x2 = x1 + w
                y2 = y1 + h

                px1 = int(landmarks[1])
                py1 = int(landmarks[2])
                px2 = int(landmarks[3])
                py2 = int(landmarks[4])
                px3 = int(landmarks[5])
                py3 = int(landmarks[6])
                px4 = int(landmarks[7])
                py4 = int(landmarks[8])
                px5 = int(landmarks[9])
                py5 = int(landmarks[10])

                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                x1 = int(x1 + w * 0.12)
                y1 = int(y1 + h * 0.1)
                x2 = int(x1 + w * 0.9)
                y2 = int(y1 + h * 0.85)
                w = x2 - x1
                h = y2 - y1

                cx = int(x1 + (w / 2))
                cy = int(y1 + (w / 2))

                _cx = cx + np.random.randint(-w * 0.2, w * 0.2)
                _cy = cy + np.random.randint(-h * 0.2, h * 0.2)
                _w = w + np.random.randint(-w * 0.2, w * 0.2)
                _h = h + np.random.randint(-h * 0.2, h * 0.2)
                _x1 = int(_cx - (_w / 2))
                _y1 = int(_cy - (_h / 2))
                _x2 = int(_x1 + _w)
                _y2 = int(_y1 + _h)

                _x1_off = (_x1 - x1) / _w
                _y1_off = (_y1 - y1) / _h
                _x2_off = (_x2 - x2) / _w
                _y2_off = (_y2 - y2) / _h

                _px1_off = (px1 - _x1) / _w
                _py1_off = (py1 - _y1) / _h
                _px2_off = (px2 - _x1) / _w
                _py2_off = (py2 - _y1) / _h
                _px3_off = (px3 - _x1) / _w
                _py3_off = (py3 - _y1) / _h
                _px4_off = (px4 - _x1) / _w
                _py4_off = (py4 - _y1) / _h
                _px5_off = (px5 - _x1) / _w
                _py5_off = (py5 - _y1) / _h

                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.image_size, self.image_size))

                iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))

                if iou > 0.6 and positive_count <= 30000:
                    clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")
                    positive_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            positive_count, 1, _x1_off, _y1_off, _x2_off, _y2_off, _px1_off, _py1_off,
                            _px2_off,_py2_off,_px3_off, _py3_off, _px4_off, _py4_off, _px5_off, _py5_off
                        ))
                    positive_label_txt.flush()
                    positive_count += 1
                elif iou > 0.4 and part_count <= 30000:
                    clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                    part_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            part_count, 2, _x1_off, _y1_off, _x2_off, _y2_off, _px1_off, _py1_off,
                            _px2_off, _py2_off, _px3_off, _py3_off, _px4_off, _py4_off, _px5_off, _py5_off
                        ))
                    part_label_txt.flush()
                    part_count += 1
                elif iou < 0.3 and negative_count <= 40000:
                    clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")
                    negative_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        ))
                    negative_label_txt.flush()
                    negative_count += 1

                w, h = img.size
                _x1, _y1 = np.random.randint(0, w), np.random.randint(0, h)
                _w, _h = np.random.randint(0, w - _x1), np.random.randint(0, h - _y1)
                _x2, _y2 = _x1 + _w, _y1 + _h
                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.image_size, self.image_size))
                iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                if iou > 0.6 and positive_count <= 30000:
                    clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")
                    positive_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            positive_count, 1, _x1_off, _y1_off, _x2_off, _y2_off, _px1_off, _py1_off,
                            _px2_off, _py2_off, _px3_off, _py3_off, _px4_off, _py4_off, _px5_off, _py5_off
                        ))
                    positive_label_txt.flush()
                    positive_count += 1
                elif iou > 0.4 and part_count <= 30000:
                    clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                    part_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            part_count, 2, _x1_off, _y1_off, _x2_off, _y2_off, _px1_off, _py1_off,
                            _px2_off, _py2_off, _px3_off, _py3_off, _px4_off, _py4_off, _px5_off, _py5_off
                        ))
                    part_label_txt.flush()
                    part_count += 1
                elif iou < 0.3 and negative_count <= 40000:
                    clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")
                    negative_label_txt.write(
                        "{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        ))
                    negative_label_txt.flush()
                    negative_count += 1

        positive_label_txt.close()
        negative_label_txt.close()
        part_label_txt.close()


if __name__ == '__main__':
    dst_path = "/Users/karson/Downloads/Dataset/"
    path = r"/Users/karson/Downloads/CelebA"
    gendata = GenData(path, dst_path, image_size=48)
    gendata.run(1)

