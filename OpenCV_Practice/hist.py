import cv2
import matplotlib.pyplot as plt
img = cv2.imread("1.jpg")

img_B = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(img_B,label="b",color="b")
img_G = cv2.calcHist([img],[1],None,[256],[0,256])
plt.plot(img_G,label="g",color="g")
img_R = cv2.calcHist([img],[2],None,[256],[0,256])
plt.plot(img_R,label="r",color="r")
plt.show()