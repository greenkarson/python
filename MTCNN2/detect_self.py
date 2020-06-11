import torch,os
import Net
from PIL import Image,ImageDraw
from torchvision import transforms
from tools import utils

tf = transforms.Compose([transforms.ToTensor()])


class Detect:
    def __init__(self):
        self.pnet = Net.PNet()
        self.pnet.load_state_dict(torch.load("pnet.pt"))
        self.pnet.eval()

        self.rnet = Net.RNet()
        # self.rnet.load_state_dict(torch.load("rnet.pt"))

        self.onet = Net.ONet()
        # self.onet.load_state_dict(torch.load("onet.pt"))

    def __call__(self, imgs):

        boxes = self.detect_pnet(imgs)

        if boxes is None:
            return []

        # boxes = self.detect_rnet(imgs, boxes)
        # if boxes is None:
        #     return []

        # boxes = self.detect_onet(imgs, boxes)
        # if boxes is None:
        #     return []
        return boxes

    def detect_pnet(self, imgs):

        scale = 1
        img_scale = imgs
        w, h = imgs.size
        res_boxes = []
        min_side = min(w, h)
        while min_side > 12:
            img_scale_tensor = tf(img_scale)

            # img_scale_tensor = img_scale_tensor[None,...]
            img_scale_tensor = torch.unsqueeze(img_scale_tensor, 0)

            predict_boxes = self.pnet(img_scale_tensor)

            predict_boxes.cpu().detach()
            torch.sigmoid_(predict_boxes[:, 0, ...])

            feature_map = predict_boxes[0, 0]
            cls_mask = feature_map > 0.7
            idx = cls_mask.nonzero()

            predict_x1 = idx[:, 1] * 2
            predict_y1 = idx[:, 0] * 2
            predict_x2 = predict_x1 + 12
            predict_y2 = predict_y1 + 12

            offset = predict_boxes[0, 1:5, cls_mask]

            x1 = (predict_x1 - offset[0, :] * 12) / scale
            y1 = (predict_y1 - offset[1, :] * 12) / scale
            x2 = (predict_x2 - offset[2, :] * 12) / scale
            y2 = (predict_y2 - offset[3, :] * 12) / scale

            cls = predict_boxes[0, 0, cls_mask]
            res_boxes.append(torch.stack([x1, y1, x2, y2, cls], dim=1))

            scale *= 0.702
            w, h = int(w * scale), int(h * scale)
            img_scale = img_scale.resize((w, h))
            print(w,h)
            min_side = min(w, h)


        # 每层产生的边框进行总拼接
        total_boxes = torch.cat(res_boxes, dim=0)
        ret = utils.nms(total_boxes.cpu().detach().numpy(),0.3)
        return ret

    def detect_rnet(self, imgs, boxes):
        pass

    def detect_onet(self, imgs, boxes):
        pass


if __name__ == "__main__":
    test_img = Image.open("2.jpg")
    img_draw = ImageDraw.Draw(test_img)
    detector = Detect()
    box = detector(test_img)

    for i in box:  # 多个框，没循环一次框一个人脸
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[2])
        y2 = int(i[3])

        # print((x1, y1, x2, y2))
        # print("conf:", i[4])  # 置信度
        img_draw.rectangle((x1, y1, x2, y2), outline='red',width=2)
    test_img.show()  # 每循环一次框一个人脸






