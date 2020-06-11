from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

 # 用PIL中的Image.open打开图像转灰度
image_arr = np.array(Image.open("9.jpg").convert('L'), 'f') # 转化成numpy数组

f = np.fft.fft2(image_arr) #傅里叶变换
fshift = np.fft.fftshift(f) #把中点移动到中间去

magnitude_spectrum = 20 * np.log(np.abs(fshift)) #计算每个频率的成分多少

plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(image_arr, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

#去掉低频信号，留下高频信号
rows, cols = image_arr.shape
crow, ccol = rows // 2, cols // 2
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0


#傅里叶逆变换
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(223), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()