import cv2

img = cv2.imread("33.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kb = orb.detect(gray)
kps,des = orb.compute(gray, kb)

dst = cv2.drawKeypoints(img, kps, None, (0,0,255),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("dst", dst)
cv2.waitKey(0)