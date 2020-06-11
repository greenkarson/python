import cv2
img = cv2.imread("1.jpg")
# dst = cv2.resize(img,(300,300))
# dst = cv2.transpose(img)
dst = cv2.flip(img,1) # 0,1,-1 三种值
cv2.imshow("dst",dst)
cv2.waitKey(0)
