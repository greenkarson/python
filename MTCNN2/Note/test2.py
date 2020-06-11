import os
import numpy as np
landmarks_path = "/Users/karson/Downloads/CelebaA/Anno/list_landmarks_celeba_test.txt"
bbox_path = "/Users/karson/Downloads/CelebaA/Anno/list_bbox_celeba_test.txt"

f_box_anno = open(bbox_path)
f_landmarks_anno = open(landmarks_path)
for i, (f_box_line, f_landmarks_line) in enumerate(zip(f_box_anno, f_landmarks_anno)):
            if i < 2:  # skip the top two lines in anno files
                continue
            image_name = f_box_line.strip().split()[0]

            boxes = f_box_line.strip().split()[1:]
            # boxes = list(filter(lambda x: x != '', boxes))
            # boxes = np.array(boxes).astype(int)
            print(boxes)

            landmarks = f_landmarks_line.strip().split()[1:]
            # landmarks = list(filter(lambda x: x != '', landmarks))
            # landmarks = np.array(landmarks).astype(int)
            print(landmarks)

