import cv2

img = cv2.imread("33.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(threshold=35)
keypiont = fast.detect(gray)

dst = cv2.drawKeypoints(img, keypiont, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("dst", dst)
cv2.waitKey()