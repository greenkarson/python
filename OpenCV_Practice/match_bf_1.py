import cv2

img1 = cv2.imread("33.jpg")
img2 = cv2.imread("34.jpg")

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray_img1, None)
kp2, des2 = orb.detectAndCompute(gray_img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des1, des2)

# lambda
matches = sorted(matches, key=lambda x: x.distance)

dst = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

cv2.imshow("img",dst)
cv2.waitKey(0)