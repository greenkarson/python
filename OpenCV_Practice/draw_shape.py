import numpy as np
import cv2

img = cv2.imread("1.jpg")
cv2.line(img,(20,100),(100,100),[0,0,255],thickness=1)
cv2.circle(img,(50,50),10,[0,0,255],thickness=2)
# 画矩形
cv2.rectangle(img,(100,100),(200,200),[0,0,255],thickness=1)
# 画椭圆
cv2.ellipse(img,(150,150),(100,50),0,0,360,[0,0,255],thickness=1)
# 画多边形
pts = np.array([[10,5],[50,10],[70,20],[20,30]],dtype=np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,0,255),thickness=2)

cv2.putText(img,"girl",(180,100),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),lineType=cv2.LINE_AA)

cv2.imshow("src",img)
cv2.waitKey(0)