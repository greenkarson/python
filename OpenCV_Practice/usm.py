import cv2

img = cv2.imread("1.jpg")
dst = cv2.GaussianBlur(img,(5,5),1)
cv2.addWeighted(img,2,dst,-1,0,dst)

cv2.imshow("src",img)
cv2.imshow("dst",img)
cv2.waitKey(0)
