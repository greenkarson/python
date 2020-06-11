import cv2

img = cv2.imread("1.jpg",0)
dst = cv2.GaussianBlur(img,(3,3),0)
dst = cv2.Canny(dst,50,150)

cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)