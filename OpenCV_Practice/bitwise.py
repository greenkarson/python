import cv2

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("666.jpg")

rows,cols,chanels = img2.shape
roi = img1[0:rows,0:cols]
# 转灰度
img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
# 二值化
ret, mask = cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 设置掩码白底黑字
mask_inv = cv2.bitwise_not(mask)

# 按位与运算
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

# cv2.imshow("...",roi)
# cv2.imshow("img2",img2)
# cv2.imshow("img2gray",img2gray)
# cv2.imshow("mask",mask)
# cv2.imshow("maks_inv",mask_inv)
# cv2.imshow("img1_bg",img1_bg)
# cv2.imshow("img2_fg",img2_fg)
cv2.imshow("dst",img1)

cv2.waitKey(0)