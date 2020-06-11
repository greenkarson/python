import cv2
src = cv2.imread("1.jpg")
# dst = cv2.cvtColor(src,cv2.COLOR_RGBA2GRAY)
dst = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

# dst = cv2.convertScaleAbs(src,alpha=6,beta=1)
cv2.imshow("src show",src)
cv2.imshow("dst show",dst)
cv2.waitKey(0)