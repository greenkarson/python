import cv2
import numpy as np
# 选取需要的区域图片
roi = cv2.imread("10.jpg")
# 读取原图
target = cv2.imread("9.jpg")
# 将图片BGR转为HSV
roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

# 计算roi直方图并进行归一化
roi_hist = cv2.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# 把roi直方图反向投影到原图上
dst = cv2.calcBackProject([target_hsv],[0,1],roi_hist,[0,180,0,256],1)

# 把零散的点连成一片
dst_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,dst_kernel)

# 把dst转为二值化图片
ret,thresh = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)

# 把图片转换成三通道
thresh = cv2.merge((thresh,thresh,thresh))
# 把原图与遮挡按位与运算合成出提取的颜色轮廓
res = cv2.bitwise_and(target,thresh)
# 结果进行拼接
res = np.hstack((target,thresh,res))
cv2.imshow("res",res)
cv2.waitKey(0)