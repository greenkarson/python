import cv2
img = cv2.imread("15.jpg",0)
ret, thres = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
contours,_ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


M = cv2.moments(contours[0])
cx,cy = int(M['m10']/M['m00']),int(M['m01']/M['m00'])
print("重心",cx,cy)

area = cv2.contourArea(contours[0])
print("面积",area)

perimeter = cv2.arcLength(contours[0], True)
print("周长:", perimeter)

