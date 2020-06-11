import cv2
import numpy as np
img = cv2.imread("27.jpg")

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img_gray,50,100)

"""
image 输入图像 
rho 步长为1像素
theta 角度步长pi/180
threshold 线段阀值超过多少是为新的直线
lines=None, 
minLineLength= 线的最短长度，比这个短的都被忽略 
maxLineGap= 两条直线之间的最大间隔，小于此值，认为是一条直线
输出上也变了，不再是直线参数的，这个函数输出的直接就是直线点的坐标位置
"""

lines = cv2.HoughLinesP(canny,1,np.pi/180,30,None,50,10)
# 提取为二维
line = lines[:,0,:]
for x1,y1,x2,y2 in line[:]:
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


"""   
image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None


lines = cv2.HoughLines(canny, 1, np.pi / 180, 100)
# 极坐标转换
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
    y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
    x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
    y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
"""


cv2.imshow("...",img)
cv2.waitKey(0)