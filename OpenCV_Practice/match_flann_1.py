import cv2
import numpy as np

# 最小匹配数量设为10个， 大于这个数量从中筛选出10个最好的
MIN_MATCH_COUNT = 10

img1 = cv2.imread('34.jpg')
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# grayImg1 = np.float32(grayImg1)
img2 = cv2.imread('33.jpg')
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# grayImg2 = np.float32(grayImg2)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(grayImg1, None)
kp2, des2 = orb.detectAndCompute(grayImg2, None)

# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
# matches = matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)

# FLANN_INDEX_KDTREE=0
# indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
# searchParams= dict(checks=50)
# flann=cv2.FlannBasedMatcher(indexParams,searchParams)
flann = cv2.FlannBasedMatcher()
# 描述文件必须为numpy float32格式
matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)


matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

cv2.imshow("img", img3)
cv2.waitKey(0)
