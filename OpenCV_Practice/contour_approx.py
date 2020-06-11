import cv2
img = cv2.imread("26.jpg")
dst = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret,thr = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

epsilon = 40 #精度
approx = cv2.approxPolyDP(contours[0],epsilon,True)

img_contour= cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)

cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)