import cv2
import numpy as np

img = cv2.imread("33.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.FastFeatureDetector_create()

cv2.drawKeypoints()