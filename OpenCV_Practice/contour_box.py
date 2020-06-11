import cv2
import numpy as np
img = cv2.imread("15.jpg")
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(dst,55,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

x,y,w,h = cv2.boundingRect(contours[0])
img_contours = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))

rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
img_contours = cv2.drawContours(img,[box],-1,(0,255,0),2)

(x,y),r = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
r = int(r)
img_contours = cv2.circle(img,center,r,(255,0,0),2)

cv2.imshow("img",img_contours)
cv2.waitKey(0)