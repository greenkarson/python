import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[3],[4]])
print(a+b)

c = np.array([5])
print(np.tile(c,[3,3]))

d = np.array([[3,4]])
print(np.tile(d,[3,1]),np.tile(d,[3]))