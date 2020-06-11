import cv2
import matplotlib.pyplot as plt
img = cv2.imread("7.jpg",0)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist,label = "hist",color="r")

dst = cv2.equalizeHist(img)
hist_eq = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.plot(hist_eq,label = "hist_eq",color="b")
plt.show()
cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)

