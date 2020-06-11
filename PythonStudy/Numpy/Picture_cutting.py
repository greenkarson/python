import numpy as np
from PIL import Image
img = Image.open("1.jpg")
#img.show()

img_data = np.array(img)
print(img_data.shape)
img_data = img_data.reshape(2, 690 // 2, 2, 750 // 2, 3)
print(img_data.shape)
img_data = img_data.transpose(0,2,1,3,4)
print(img_data.shape)
img_data = img_data.reshape(4,345,375,3)
print(img_data.shape)
imgs = np.split(img_data,4)

print(imgs[0].shape)

# for i in imgs:
#     img = Image.fromarray(i[0])
#     img.show()
img_0 = imgs[0][0]
img_1 = imgs[1][0]
img_2 = imgs[2][0]
img_3 = imgs[3][0]
# imgs_data = np.concatenate([img_0[None,...],img_1[None,...],img_2[None,...],img_3[None,...]])
imgs_data = np.stack([img_0,img_1,img_2,img_3])
imgs_data = imgs_data.reshape(2,2,345,375,3)
imgs_data = imgs_data.transpose(0,2,1,3,4)
imgs_data = imgs_data.reshape(690,750,3)
a = Image.fromarray(imgs_data)
a.show()
print(imgs_data.shape)
