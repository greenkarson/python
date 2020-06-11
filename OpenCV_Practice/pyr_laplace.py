import cv2
img = cv2.imread("12.jpg")
img_down = cv2.pyrDown(img)
img_up = cv2.pyrUp(img_down)

img_new = cv2.subtract(img, img_up)
img_new =cv2.convertScaleAbs(img_new,alpha=5)

cv2.imshow("new",img_new)
cv2.waitKey(0)