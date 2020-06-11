from Net import *
from data import tf
from PIL import Image, ImageDraw
from tools import utils
import numpy as np


class Detector:

    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("param/pnet2020-04-19-00-09-26.pt"))
        self.pnet.eval()

        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load("param/rnet2020-04-19-08-48-18.pt"))
        self.rnet.eval()

        self.onet = ONet()
        self.onet.load_state_dict(torch.load("param/onet2020-04-18-09-27-42.pt"))
        self.onet.eval()

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes is None:
            return []

        print(boxes.shape)
        boxes = self.detRnet(img, boxes)
        if boxes is None:
            return []
        print(boxes.shape)

        boxes = self.detOnet(img, boxes)
        if boxes is None:
            return []
        print(boxes.shape)
        return boxes

    def detPnet(self, img):
        w, h = img.size
        scale = 1
        img_scale = img

        min_side = min(w, h)

        _boxes = []
        while min_side > 12:
            _img_scale = tf(img_scale)
            y = self.pnet(_img_scale[None, ...])
            y = y.cpu().detach()

            torch.sigmoid_(y[:, 0, ...])
            c = y[0, 0]
            c_mask = c > 0.48
            idxs = c_mask.nonzero()
            _x1, _y1 = idxs[:, 1] * 2, idxs[:, 0] * 2  # 2为整个P网络代表的步长
            _x2, _y2 = _x1 + 12, _y1 + 12

            p = y[0, 1:, c_mask]
            x1 = (_x1 - p[0, :] * 12) / scale
            y1 = (_y1 - p[1, :] * 12) / scale
            x2 = (_x2 - p[2, :] * 12) / scale
            y2 = (_y2 - p[3, :] * 12) / scale

            cc = y[0, 0, c_mask]

            _boxes.append(torch.stack([x1, y1, x2, y2, cc], dim=1))

            # 图像金字塔
            scale *= 0.702
            _w, _h = int(w * scale), int(h * scale)
            img_scale = img_scale.resize((_w, _h))
            min_side = min(_w, _h)

        boxes = torch.cat(_boxes, dim=0)
        return utils.nms(boxes.cpu().detach().numpy(), 0.33)

    def detRnet(self, img, boxes):
        imgs = []
        for box in boxes:
            crop_img = img.crop(box[0:4])
            crop_img = crop_img.resize((24, 24))
            imgs.append(tf(crop_img))
        _imgs = torch.stack(imgs, dim=0)

        y = self.rnet(_imgs)

        y = y.cpu().detach()
        torch.sigmoid_(y[:, 0])
        y = y.numpy()
        # print(y[:,0])

        c_mask = y[:, 0] > 0.55
        _boxes = boxes[c_mask]
        print(_boxes.shape)

        _y = y[c_mask]

        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1 = _boxes[:, 0] - _y[:, 1] * _w
        y1 = _boxes[:, 1] - _y[:, 2] * _h
        x2 = _boxes[:, 2] - _y[:, 3] * _w
        y2 = _boxes[:, 3] - _y[:, 4] * _h
        cc = _y[:, 0]

        _boxes = np.stack([x1, y1, x2, y2, cc], axis=1)

        return utils.nms(_boxes, 0.3)

    def detOnet(self, img, boxes):
        imgs = []
        for box in boxes:
            crop_img = img.crop(box[0:4])
            crop_img = crop_img.resize((48, 48))
            imgs.append(tf(crop_img))
        _imgs = torch.stack(imgs, dim=0)

        y = self.onet(_imgs)
        y = y.cpu().detach()
        torch.sigmoid_(y[:, 0])
        y = y.numpy()

        c_mask = y[:, 0] > 0.7
        _boxes = boxes[c_mask]
        print(_boxes.shape)

        _y = y[c_mask]

        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1 = _boxes[:, 0] - _y[:, 1] * _w
        y1 = _boxes[:, 1] - _y[:, 2] * _h
        x2 = _boxes[:, 2] - _y[:, 3] * _w
        y2 = _boxes[:, 3] - _y[:, 4] * _h
        cc = _y[:, 0]

        # 生成数据错误
        landmarks = _y[:, 5:]
        px1 = landmarks[:, 0] * _w + _boxes[:, 0]
        py1 = landmarks[:, 1] * _h + _boxes[:, 1]
        px2 = landmarks[:, 2] * _w + _boxes[:, 0]
        py2 = landmarks[:, 3] * _h + _boxes[:, 1]
        px3 = landmarks[:, 4] * _w + _boxes[:, 0]
        py3 = landmarks[:, 5] * _h + _boxes[:, 1]
        px4 = landmarks[:, 6] * _w + _boxes[:, 0]
        py4 = landmarks[:, 7] * _h + _boxes[:, 1]
        px5 = landmarks[:, 8] * _w + _boxes[:, 0]
        py5 = landmarks[:, 9] * _h + _boxes[:, 1]

        _boxes = np.stack([x1, y1, x2, y2, cc, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], axis=1)

        _boxes = utils.nms(_boxes, 0.3)
        _boxes = utils.nms(_boxes, 0.3, is_min=True)
        return _boxes


if __name__ == '__main__':
    test_img = Image.open("12.jpg")
    img_draw = ImageDraw.Draw(test_img)
    detector = Detector()
    box = detector(test_img)

    for i in box:  # 多个框，没循环一次框一个人脸
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[2])
        y2 = int(i[3])

        # px1 = int(i[5])
        # py1 = int(i[6])
        # px2 = int(i[7])
        # py2 = int(i[8])
        # px3 = int(i[9])
        # py3 = int(i[10])
        # px4 = int(i[11])
        # py4 = int(i[12])
        # px5 = int(i[13])
        # py5 = int(i[14])

        # print((x1, y1, x2, y2))
        # print("conf:", i[4])  # 置信度
        img_draw.rectangle((x1, y1, x2, y2), outline='green', width=2)

        # img_draw.point((px1, py1),fill="green")
        # img_draw.point((px2, py2), fill="green")
        # img_draw.point((px3, py3), fill="green")
        # img_draw.point((px4, py4), fill="green")
        # img_draw.point((px5, py5), fill="green")

    test_img.show()  # 每循环一次框一个人脸



