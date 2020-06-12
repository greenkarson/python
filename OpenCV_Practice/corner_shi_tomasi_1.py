import cv2
import numpy as np
img = cv2.imread("32.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)

corners = cv2.goodFeaturesToTrack(gray,100, 0.01, 10)
print(corners)
corners = np.int0(corners)
# print(corners)

for i in corners:
    x, y = i.ravel()
    # print(x,y)
    cv2.circle(img, (x, y), 3, 255)

cv2.imshow("img", img)
cv2.waitKey(0)