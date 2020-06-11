import cv2
img = cv2.imread("16.jpg")
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(dst,55,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

ellipse = cv2.fitEllipse(contours[0])
cv2.ellipse(img,ellipse,(0,0,255),2)

h,w,_ = img.shape
[vx,vy,x,y] = cv2.fitLine(contours[0],cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((w - x) * vy / vx) + y)
cv2.line(img, (w - 1, righty), (0, lefty), (0, 0, 255), 2)

cv2.imshow("img_contour", img)
cv2.waitKey(0)
