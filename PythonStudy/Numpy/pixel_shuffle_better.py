import numpy as np

a = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]], [[4, 4], [4, 4]]])
print(a.shape)
a = a.reshape(2,2,2,2)
print(a.shape)

a = a.transpose(2,0,3,1)
print(a.shape)
print(a)
a = a.reshape(4,4)
print(a)