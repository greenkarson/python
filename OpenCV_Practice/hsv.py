import numpy as np
import cv2

img = cv2.imread("11.jpg")
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
lower_blue = np.array([100,200,100])
upper_blue = np.array([200,255,200])
maks = cv2.inRange(hsv,lower_blue,upper_blue)
res = cv2.bitwise_and(img,img,mask=maks)
cv2.imshow("src",img)
cv2.imshow("maks",maks)
cv2.waitKey(0)