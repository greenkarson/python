import numpy as np

# 11. 如何获取两个numpy数组之间的公共项？
# **难度等级：**L2
# **问题：**获取数组a和数组b之间的公共项。
# 给定：
# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 期望的输出：
# array([2, 4])

# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# print(np.intersect1d(a,b))

# #12. 如何从一个数组中删除存在于另一个数组中的项？
# **难度等级：**L2
# **问题：**从数组a中删除数组b中的所有项。
# 给定：
# a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 期望的输出：
# array([1,2,3,4])

# a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# print(np.setdiff1d(a,b))
#13. 如何得到两个数组元素匹配的位置？
# **难度等级：**L2
# **问题：**获取a和b元素匹配的位置。
# 给定：
# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 期望的输出：
# > (array([1, 3, 5, 7]),)

# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# print(np.where(a == b))

#14. 如何从numpy数组中提取给定范围内的所有数字？
# **难度等级：**L2
# **问题：**获取5到10之间的所有项目。
# 给定：
# a = np.array([2, 6, 1, 9, 10, 3, 27])
# 期望的输出：
# (array([6, 9, 10]),)

a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >= 5) & (a <= 10))
print(a[index])

#15. 如何创建一个python函数来处理scalars并在numpy数组上工作？
# **难度等级：**L2
# **问题：**转换适用于两个标量的函数maxx，以处理两个数组。
# 给定：
# def maxx(x, y):
#     """Get the maximum of two items"""
#     if x >= y:
#         return x
#     else:
#         return y
# maxx(1, 5)
#  > 5
# 期望的输出：
# a = np.array([5, 7, 9, 8, 6, 4, 5])
# b = np.array([6, 3, 4, 8, 9, 7, 1])
# pair_max(a, b)
# > array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
pair_max = np.vectorize(maxx,otypes=[float])
print(pair_max(a,b))

#16. 如何交换二维numpy数组中的两列？
# **难度等级：**L2
# **问题：**在数组arr中交换列1和2。
# 给定：
# arr = np.arange(9).reshape(3,3)

arr = np.arange(9).reshape(3,3)
print(arr[[0,2,1],:])

#17. 如何交换二维numpy数组中的两行？
# **难度等级：**L2
# **问题：**交换数组arr中的第1和第2行：
# 给定：
# arr = np.arange(9).reshape(3,3)

print(arr[:,[0,2,1]])


#18. 如何反转二维数组的行？
# **难度等级：**L2
# **问题：**反转二维数组arr的行。
# 给定：
# Input
# arr = np.arange(9).reshape(3,3)

print(arr[::-1])

#19. 如何反转二维数组的列？
# **难度等级：**L2
# **问题：**反转二维数组arr的列。
# 给定：
# Input
# arr = np.arange(9).reshape(3,3)

print(arr[:,::-1])

#20. 如何创建包含5到10之间随机浮动的二维数组？
# **难度等级：**L2
# **问题：**创建一个形状为5x3的二维数组，以包含5到10之间的随机十进制数。



#21. 如何在numpy数组中只打印小数点后三位？
# **难度等级：**L1
# **问题：**只打印或显示numpy数组rand_arr的小数点后3位。
# 给定：
# rand_arr = np.random.random((5,3))
