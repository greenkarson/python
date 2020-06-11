import cv2
from PIL import Image
path = "/Users/karson/Downloads/CelebaA/img_celeba.7z/000001.jpg"
src_label = "/Users/karson/Downloads/CelebaA/Anno/list_bbox_celeba.txt"

img = cv2.imread(path, 0)
img = cv2.convertScaleAbs(img, alpha=1, beta=0)
img = cv2.GaussianBlur(img, (3, 3), 1)
dst = cv2.Canny(img, 50, 150)

x1, y1, w, h = 95,71,226,313
x2, y2 = x1 + w, y1 + h
cv2.rectangle(dst,(x1,y1),(x2,y2),[0,0,255],thickness=3)

cv2.imshow("...", img)
cv2.imshow("dst", dst)
cv2.waitKey(0)


