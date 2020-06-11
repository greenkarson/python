import cv2
img = cv2.imread("13.jpg")

for i in range(3):
    cv2.imshow(f"img{i}",img)
    img = cv2.pyrUp(img)
    # img = cv2.pyrDown(img)

cv2.waitKey(0)