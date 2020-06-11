import os, cv2
import numpy as np
from PIL import Image, ImageDraw

src_path = r"/Users/karson/Downloads/CelebaA/img_celeba"
src_label = r"/Users/karson/Downloads/CelebaA/Anno/list_bbox_celeba.txt"
test_path = "/Users/karson/Downloads/Test"

path = "/Users/karson/Downloads/CelebaA/img_celeba.7z/000001.jpg"

for i, line in enumerate(open(src_label, "r")):
    strs = line.split()
    if i < 2:
        continue
    img_path = f"{src_path}/{strs[0]}"
    img = Image.open(img_path)
    x1, y1, w, h = int(strs[1]), int(strs[2]), int(strs[3]), int(strs[4])
    x2, y2 = x1 + w, y1 + h
    print(x1, y1, x2, y2)
    img_draw = ImageDraw.Draw(img)
    img_draw.rectangle((x1, y1, x2, y2), outline="green", width=2)
    # img.save(f"/Users/karson/Downloads/Test/{i}.jpg")

    # img_crop = img.crop([x1,y1,x2,y2])
    # img_crop.save(f"/Users/karson/Downloads/Test/{i}.jpg")
    _x1 = int(x1 + w * 0.12)
    _y1 = int(y1 + h * 0.1)
    _x2 = int(x1 + w * 0.9)
    _y2 = int(y1 + h * 0.85)
    img_draw.rectangle((_x1, _y1, _x2, _y2), outline="red", width=2)
    img.save(f"/Users/karson/Downloads/Test/{i}.jpg")
