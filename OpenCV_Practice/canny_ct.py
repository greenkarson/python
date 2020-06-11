import cv2

img = cv2.imread("25.jpg")
# 对比度增强去噪
img = cv2.convertScaleAbs(img,alpha=6,beta=0)
img = cv2.GaussianBlur(img,(3,3),1)
# 轮廓提取补洞
dst = cv2.Canny(img,50,150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dst2 = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel)
# dst = cv2.resize(dst,(500,500))

cv2.imshow("src show",img)
cv2.imshow("dst show",dst)

cv2.waitKey(0)