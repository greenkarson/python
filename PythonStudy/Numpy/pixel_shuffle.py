import numpy as np

a = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]], [[4, 4], [4, 4]]])
a = a.reshape(1, 4, 4)
a = a.transpose(0, 2, 1)
a = a.reshape(4, 2, 2)
a_data = np.split(a, 4)

a_0 = a_data[0][0]
a_1 = a_data[1][0]
a_2 = a_data[2][0]
a_3 = a_data[3][0]

b = np.stack([a_0, a_1], axis=1)
b = b.reshape(1, 2, 4)
c = np.stack([a_2, a_3], axis=1)
c = c.reshape(1, 2, 4)
d = np.stack([b, c])
d = d.transpose(1, 0, 2, 3)
d = d.reshape(1, 4, 4)
print(d)
