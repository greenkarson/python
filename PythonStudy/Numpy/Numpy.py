import numpy as np

# 创建数组
# a = np.array([[1, 2], [3, 4]])
# i = np.arange(0, 12)
# 创建高斯分布采样的随机数组
# b = np.random.rand(3, 4)
# 创建均匀分布采样的随机数组
# d = np.random.uniform(0, 1, (3, 4))
# uninitialized output
# c = np.empty((3, 3))
# 创建连续数组,并转型
# e = np.arange(0, 12).reshape(3, 4)
# 创建数组范围0-2均分为12份
# f = np.linspace(0, 2, 12)

# nidm数组轴维度个数
# shape数组的形状
# size数组大小元素个数
# dtype数组类型，如整型浮点型

# print(a.ndim,a.shape,a.size,a.dtype)

# 数据类型转化
# a = np.array([[1., 2.], [3., 4.]],dtype=float32)

# 数组数据运算
# a = a ** 2
# a = 10*np.sin(a)

# 数组截取操作>,可以截取元素
# b = b > 2
# print(i[i > 5])

# 聚合操作
# i = np.arange(0, 12)
# print(i.sum(), i.max(), i.mean(), i.min())

# 多维度操作按轴相加，按照形状意义来进行划分
# e = np.arange(0, 12).reshape(3, 4, 5)
# print(e.sum(axis=1))

# 最大值索引返回
# e = np.arange(0, 12).reshape(3, 4)
# print(e.argmax(axis=1))
# print(e[e>6],np.where(e>6))

# 累加之前元素
# print(e.cumsum())

# 通函数，所有元素作用

# 索引、切片、迭代
# a = np.array([[1, 2], [3, 4]])
# 单个元素索引
# print(a[1, 0])
# 多个元素索引, :冒号表示维度全部取
# print(a[:,1])

# 索引
# k = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 起始索引:结束索引:步长 ,[] 表示纬度
# print(k[0:2,3])
# 0可省略,纬度最大值可省略
# print(k[0:3:2,2])
# print(k[:3:2,2])
# print(k[::2,2])
# print(k[[0,2],2])

# print(k[[0,2],0:2])

# 索引简写
# m = np.random.rand(2,3,4,5)
# print(m[:,:,:,1])
# print(m[...,1])



# 增加纬度方法
# o = np.arange(0, 12)
# print(o.reshape(3, 4))
# print(o.reshape(-1,4))
# print(o[:,None])
# print(o.squeeze())

# 减纬度方法
# p = np.arange(0,12).reshape(3,1,4)
# print(p[:,0,:])

# 多维度数据展平
# p = np.arange(0,12).reshape(3,1,4)
# print(p.flatten())

# 拼接
# a = np.array([[1,2,3],[2,3,4]])
# b = np.array([[5,6,4],[7,8,9]])
# 按纬度拼接
# c = np.stack([a,b],axis=1)
# print(c.shape)

#内部拼接
# a = np.array([[1,2,3],[2,3,4]])
# b = np.array([[5,6,4],[7,8,9]])
# c = np.concatenate([a,b],axis=1)
# print(c)



