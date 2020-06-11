import cv2
import numpy as np
img = cv2.imread("30.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

# 去除噪音
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
# 背景分离
surge_bg =cv2.dilate(opening,kernel,iterations=3)
# 形成山峰
dist_transform = cv2.distanceTransform(opening,1,5)
ret,surge_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,cv2.THRESH_BINARY)
# 找到未知区域
surge_fg = np.uint8(surge_fg)
unknown = cv2.subtract(surge_bg,surge_fg)
# 寻找中心
ret,marker1 = cv2.connectedComponents(surge_fg)
markers = marker1 + 1
markers[unknown == 255] =0

markers3 = cv2.watershed(img,markers)
img[markers3 == -1] = [0,0,255]
cv2.imshow("...",unknown)
cv2.waitKey(0)

