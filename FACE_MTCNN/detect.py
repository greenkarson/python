from Network import PNet, RNet, ONet
from PIL import Image, ImageDraw
from torchvision import transforms
import torch, math
from tools.utils import nms
from tools.utils import old_nms
import numpy as np


class FaceDetector():
    def __init__(self):

        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("./param/pnet.pt"))
        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load("./param/rnet.pt"))
        self.onet = ONet()
        self.onet.load_state_dict(torch.load("./param/onet.pt"))

    def detect(self, img):
        width, height = img.size
        scale = 1
        scale_img = img
        # print(scale_img.shape)
        min_side = min(width, height)
        # print(min_side)
        _boxes = []
        while min_side > 12:
            tf_img = self.img_preprocess(scale_img)
            cls, box_regs, _ = self.pnet(tf_img)
            cls = cls.cpu().detach()
            box_regs = box_regs.cpu().detach()
            probs = cls[0, 1, :, :]
            # print(probs.shape, box_regs.shape)
            probs_mask = probs > 0.4
            # print(probs_mask.shape)

            index = probs_mask.nonzero()
            # print(index.shape)
            tx1, ty1 = index[:, 1] * 2, index[:, 0] * 2
            tx2, ty2 = tx1 + 12, ty1 + 12
            # print(tx1.shape,ty1.shape,tx2.shape, ty2.shape)
            offset_mask = box_regs[0, :, probs_mask]
            # print(offset_mask.shape)
            x1 = (tx1 + offset_mask[0, :] * 12) / scale
            y1 = (ty1 + offset_mask[1, :] * 12) / scale
            x2 = (tx2 + offset_mask[2, :] * 12) / scale
            y2 = (ty2 + offset_mask[3, :] * 12) / scale
            # print(x1.shape, y1.shape, x2.shape, y2.shape)

            score = probs[probs_mask]
            # print(score.shape)
            _boxes.append(torch.stack([x1, y1, x2, y2, score], dim=1))

            scale *= 0.702
            cur_width, cur_height = int(width * scale), int(height * scale)
            print(cur_width, cur_height)
            scale_img = scale_img.resize((cur_width, cur_height))

            min_side = min(cur_width, cur_height)

        boxes = torch.cat(_boxes, dim=0)
        print(boxes.shape)
        return old_nms(boxes.cpu().detach().numpy(), 0.3)
        # return boxes

    def pnet_detect(self, img):
        scale_img = img
        # print(scale_img.shape)
        width = scale_img.shape[3]
        height = scale_img.shape[2]

        scales = []
        cur_width = width
        cur_height = height
        cur_factor = 1

        while cur_width >= 12 and cur_height >= 12:
            if 12 / cur_factor >= 12:
                w = cur_width
                h = cur_height
                scales.append((w, h, cur_factor))
                # print(w,h,cur_factor)

            cur_factor *= 0.7
            cur_width = math.ceil(cur_width * 0.7)
            cur_height = math.ceil(cur_height * 0.7)

        candidate_boxes = torch.empty((0, 4))
        candidate_score = torch.empty((0))
        candidate_offsets = torch.empty((0, 4))

        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(scale_img, size=(h, w), mode="bilinear", align_corners=True)

            # print(resize_img.shape,f)
            p_distribution, box_regs, _ = self.pnet(resize_img)
            # print(p_distribution.shape, box_regs.shape)
            p_distribution = p_distribution.cpu().detach()
            box_regs = box_regs.cpu().detach()
            # print(p_distribution[0,1,:,:],p_distribution.shape, box_regs.shape)

            candidate, scores, offsets = self.generate_bboxes(p_distribution, box_regs, f)
            candidate = candidate.float()
            # print(candidate.shape, scores.shape, offsets.shape)

            candidate_boxes = torch.cat([candidate_boxes, candidate])
            candidate_score = torch.cat([candidate_score, scores])
            candidate_offsets = torch.cat([candidate_offsets, offsets])
            # print(candidate.shape, scores.shape, offsets.shape)

        if candidate_boxes.shape[0] != 0:
            # candidate_boxes = self.calibrate_box(candidate_boxes, candidate_offsets)
            keep = nms(candidate_boxes.cpu().numpy(), candidate_score.cpu().numpy(), 0.3)
            return candidate_boxes[keep]
        else:
            return candidate_boxes

    def rnet_detect(self, img, boxes):
        if boxes.shape[0] == 0:
            return boxes
        width, height = img.size
        boxes = self.convert_to_square(boxes)
        # boxes = self.refine_boxes(boxes, width, height)

        candidate_faces = list()
        for box in np.array(boxes):
            # im = img[:, :, box[1]: box[3], box[0]: box[2]]
            # im = torch.nn.functional.interpolate(im, size=(24, 24), mode='bilinear')
            # candidate_faces.append(im)

            # print(box[0], box[1], box[2], box[3])
            crop_img = img.crop(box[0:4])
            crop_img = crop_img.resize((24, 24))
            candidate_faces.append(self.img_preprocess(crop_img))

        candidate_faces = torch.cat(candidate_faces, 0)
        # print(candidate_faces.shape)

        # rnet forward pass
        p_distribution, box_regs, _ = self.rnet(candidate_faces)
        p_distribution = p_distribution.cpu().detach()
        box_regs = box_regs.cpu().detach()
        # print(p_distribution.shape, box_regs.shape)

        # filter negative boxes
        scores = p_distribution[:, 1]
        # print(scores)
        # print(scores.shape)

        mask = (scores >= 0)
        boxes = boxes[mask]
        # print(boxes.shape)
        box_regs = box_regs[mask]

        scores = scores[mask]
        # print(scores.shape)
        # c = list()
        # for i in boxes:
        #     x1, y1, x2, y2 = i[0], i[1], i[2], i[3]
        #     c.append(torch.tensor([[x1, y1, x2, y2]]))
        # c = torch.cat(c, 0)
        # cc = torch.stack([c[:,0],c[:,1],c[:,2],c[:,3],scores],1)
        # xxxx = old_nms(cc.cpu().numpy(), 0.2)
        # print(xxxx.shape)
        # return xxxx

        if boxes.shape[0] > 0:
            # boxes = self.calibrate_box(boxes, box_regs)
            # nms
            keep = nms(boxes.cpu().numpy(), scores.cpu().numpy(), 0.3)
            boxes = boxes[keep]
            print(boxes.shape)
        return boxes

    def onet_detect(self, img, boxes):
        if boxes.shape[0] == 0:
            return boxes, torch.empty(0, dtype=torch.int32)

        width, height = img.size

        boxes = self.convert_to_square(boxes)
        # boxes = self.refine_boxes(boxes, width, height)

        # get candidate faces
        candidate_faces = list()

        for box in np.array(boxes):
            # im = img[:, :, box[1]: box[3], box[0]: box[2]]
            # im = torch.nn.functional.interpolate(
            #     im, size=(48, 48), mode='bilinear')
            # candidate_faces.append(im)
            # print(box[0], box[1], box[2], box[3])

            crop_img = img.crop(box[0:4])
            crop_img = crop_img.resize((48, 48))
            candidate_faces.append(self.img_preprocess(crop_img))

        candidate_faces = torch.cat(candidate_faces, 0)
        # print(candidate_faces.shape)

        p_distribution, box_regs, landmarks = self.onet(candidate_faces)
        p_distribution = p_distribution.cpu().detach()
        box_regs = box_regs.cpu().detach()
        landmarks = landmarks.cpu().detach()
        # print(p_distribution.shape, box_regs.shape, landmarks.shape)

        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= 0.4)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if boxes.shape[0] > 0:
            # compute face landmark points
            landmarks = self.calibrate_landmarks(boxes, landmarks)
            landmarks = torch.stack([landmarks[:, :5], landmarks[:, 5:10]], 2)
            # boxes = self.calibrate_box(boxes, box_regs)
            # boxes = self.refine_boxes(boxes, width, height)

            # nms
            keep = nms(boxes.cpu().numpy(), scores.cpu().numpy(), 0.3)
            boxes = boxes[keep]
            landmarks = landmarks[keep]

        return boxes, landmarks

    def img_preprocess(self, img):
        tf = transforms.Compose([transforms.ToTensor()])
        img = tf(img)
        img = (img - 127.5) * 0.0078125
        img = img.unsqueeze(0)
        return img

    def generate_bboxes(self, probs, offsets, scale):
        stride = 2
        cell_size = 12

        cls = probs[0, 1, :, :]
        # print(probs.shape)

        inds_mask = cls > 0.5
        inds = inds_mask.nonzero()
        # print("inds:",inds.shape)
        # print(offsets.shape)

        if inds.shape[0] == 0:
            return torch.empty((0, 4)), torch.empty(0), torch.empty((0, 4))

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[:, 0], inds[:, 1]] for i in range(4)]
        # print(tx1.shape, ty1.shape, tx2.shape, ty2.shape)
        offsets = torch.stack([tx1, ty1, tx2, ty2], dim=1)
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        score = cls[inds[:, 0], inds[:, 1]]
        # print(score.shape)

        bounding_boxes = torch.stack([
            stride * inds[:, 1] + 1.0,
            stride * inds[:, 0] + 1.0,
            stride * inds[:, 1] + 1.0 + cell_size,
            stride * inds[:, 0] + 1.0 + cell_size], 0).transpose(0, 1)
        # print(bounding_boxes.shape)
        bounding_boxes = torch.round(bounding_boxes / scale).int()
        # print(bounding_boxes.shape)
        # exit()
        return bounding_boxes, score, offsets

    def calibrate_box(self, bboxes, offsets):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = torch.unsqueeze(w, 1)
        h = torch.unsqueeze(h, 1)

        translation = torch.cat([w, h, w, h], 1).float() * offsets
        bboxes += torch.round(translation).int()
        return bboxes

    def convert_to_square(self, bboxes):

        square_bboxes = torch.zeros_like(bboxes, dtype=torch.float32)
        x1, y1, x2, y2 = [bboxes[:, i].float() for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = torch.max(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0

        square_bboxes = torch.ceil(square_bboxes + 1).int()
        return square_bboxes

    def refine_boxes(self, bboxes, w, h):

        bboxes = torch.max(torch.zeros_like(bboxes), bboxes)
        sizes = torch.IntTensor([[h, w, h, w]]) * bboxes.shape[0]
        bboxes = torch.min(bboxes, sizes)
        return bboxes

    def calibrate_landmarks(self, bboxes, landmarks, align=False):

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]

        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = torch.unsqueeze(w, 1)
        h = torch.unsqueeze(h, 1)

        translation = torch.cat([w] * 5 + [h] * 5, 1).float() * landmarks

        if align:
            landmarks = torch.ceil(translation).int()
        else:
            landmarks = torch.stack([bboxes[:, 0]] * 5 + [bboxes[:, 1]] * 5, 1) + torch.round(translation).int()
        return landmarks


if __name__ == '__main__':
    img = Image.open("1.jpg")
    img_draw = ImageDraw.Draw(img)
    detect = FaceDetector()
    p_img = detect.img_preprocess(img)

    p_boxes = detect.pnet_detect(p_img)
    print(p_boxes.shape)
    r_boxes = detect.rnet_detect(img, p_boxes)
    print(r_boxes.shape)
    o_boxes, landmarks = detect.onet_detect(img, r_boxes)
    print(o_boxes.shape, landmarks.shape)

    # boxes = detect.detect(img)

    for box in r_boxes:
        # Default draw red box on it.

        img_draw.rectangle((box[0], box[1], box[2], box[3]), outline='green', width=2)

    img.show()
