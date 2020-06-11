import cv2
img = cv2.imread("2.jpg")
cv2.bilateralFilter()
dst = cv2.blur(img,(3,3))
dst = cv2.medianBlur(img,(3,3))
dst = cv2.GaussianBlur(img,(3,3),1)

dst = cv2.Laplacian(img,-1)

dst = cv2.Sobel(img,-1,1,0)
dst = cv2.Sobel(img,-1,0,1)
dst = cv2.Scharr(img,-1,1,0)

cv2.imshow("src show", img)
cv2.imshow("dst show", dst)
cv2.waitKey(0)