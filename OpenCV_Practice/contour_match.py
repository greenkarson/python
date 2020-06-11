import cv2

img = cv2.imread("16.jpg",0)
img2 = cv2.imread("17.jpg",0)

ret1,thresh1 = cv2.threshold(img,55,255,cv2.THRESH_BINARY)
contours,_ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours[0]

ret2,thresh2 = cv2.threshold(img,55,255,cv2.THRESH_BINARY)
contours,_ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1,cnt2,cv2.CONTOURS_MATCH_I2,0)
print(ret)