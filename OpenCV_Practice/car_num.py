import cv2

raw_img = cv2.imread("23.jpg")

img = cv2.GaussianBlur(raw_img,(3,3),0)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

Sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
abs_x = cv2.convertScaleAbs(Sobel_x)
ret,thresh = cv2.threshold(abs_x,100,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(17,5))
img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernelx)
cv2.imshow('image', img)
cv2.waitKey(0)
exit()
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(1,19))

img = cv2.dilate(img,kernelx)
img = cv2.erode(img,kernelx)
img = cv2.dilate(img,kernely)
img = cv2.erode(img,kernely)

image = cv2.medianBlur(img, 15)
cv2.imshow("...2.", image)
# 查找轮廓
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2):
        # 裁剪区域图片
        chepai = raw_img[y:y + height, x:x + weight]
        cv2.imshow('chepai' + str(x), chepai)

# 绘制轮廓
image = cv2.drawContours(raw_img, contours, -1, (0, 0, 255), 3)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()