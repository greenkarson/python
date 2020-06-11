import cv2
img = cv2.imread("15.jpg")
dst = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret,thres = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

hull = cv2.convexHull(contours[0])

print(cv2.isContourConvex(contours[0]),cv2.isContourConvex(hull))
# False True
# 轮廓是非凸的，凸包是凸的
img_contour = cv2.drawContours(img,[hull],-1,(0,0,255),2)

cv2.imshow("img",img_contour)
cv2.waitKey(0)