import numpy as np


def iou(box, boxes, is_min=False):
    # box = x1,y1,x2,y2
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    if is_min:
        ovr = np.true_divide((w * h), np.minimum(box_area, boxes_area))
        return ovr
    else:
        ovr = np.true_divide((w * h), (box_area + boxes_area - (w * h)))
        return ovr


def nms(boxes, threshold, is_min=False):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        # 取出第1个框
        a_box = _boxes[0]
        # 取出剩余的框
        b_boxes = _boxes[1:]

        # 将1st个框加入列表
        r_boxes.append(a_box)  # 每循环一次往，添加一个框
        _boxes = b_boxes[iou(a_box, b_boxes, is_min) < threshold]

    if _boxes.shape[0] > 0:
        # 最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；
        # ★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0])  # 将此框添加到列表中

    return np.array(r_boxes)


if __name__ == '__main__':
    b = [38, 50, 120, 180]
    bs = [[38, 50, 120, 180], [45, 56, 110, 200]]
    bs = np.array(bs)
    res = iou(b, bs)
    print(res)
