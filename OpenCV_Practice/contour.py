import cv2
img = cv2.imread("14.jpg")
dst = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
retval, dst = cv2.threshold(dst,10,255,cv2.THRESH_BINARY)

# contours,hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours,hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


img_contour = cv2.drawContours(img,contours,-1,(0,0,255),thickness=2)

cv2.imshow("contour",img_contour)
cv2.waitKey(0)

