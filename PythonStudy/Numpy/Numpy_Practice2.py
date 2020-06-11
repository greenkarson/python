# 22. 如何通过e式科学记数法（如1e10）来打印一个numpy数组？
# **难度等级：**L1
# **问题：**通过e式科学记数法来打印rand_arr（如1e10）
# 给定：
# Create the random array
# np.random.seed(100)
# rand_arr = np.random.random([3,3])/1e3
# rand_arr
# > array([[  5.434049e-04,   2.783694e-04,   4.245176e-04],
# >        [  8.447761e-04,   4.718856e-06,   1.215691e-04],
# >        [  6.707491e-04,   8.258528e-04,   1.367066e-04]])
# 期望的输出：
# > array([[ 0.000543,  0.000278,  0.000425],
# >        [ 0.000845,  0.000005,  0.000122],
# >        [ 0.000671,  0.000826,  0.000137]])



#23. 如何限制numpy数组输出中打印的项目数？
# **难度等级：**L1
# **问题：**将numpy数组a中打印的项数限制为最多6个元素。
# 给定：
# a = np.arange(15)
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
# 期望的输出：
# > array([ 0,  1,  2, ..., 12, 13, 14])


#24. 如何打印完整的numpy数组而不截断
# **难度等级：**L1
# **问题：**打印完整的numpy数组a而不截断。
# 给定：
# np.set_printoptions(threshold=6)
# a = np.arange(15)
# > array([ 0,  1,  2, ..., 12, 13, 14])
# 期望的输出：
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])


#25. 如何导入数字和文本的数据集保持文本在numpy数组中完好无损？
# **难度等级：**L2
# **问题：**导入鸢尾属植物数据集，保持文本不变。

#26. 如何从1维元组数组中提取特定列？
# **难度等级：**L2
# **问题：**从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)


#27. 如何将1维元组数组转换为2维numpy数组？
# **难度等级：**L2
# **问题：**通过省略鸢尾属植物数据集种类的文本字段，将一维鸢尾属植物数据集转换为二维数组iris_2d。
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)


#28. 如何计算numpy数组的均值，中位数，标准差？
# **难度等级：**L1
# **问题：**求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')



#29. 如何规范化数组，使数组的值正好介于0和1之间？
# **难度等级：**L2
# **问题：**创建一种标准化形式的鸢尾属植物间隔长度，其值正好介于0和1之间，这样最小值为0，最大值为1。
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])


#30. 如何计算Softmax得分？
# **难度等级：**L3
# **问题：**计算sepallength的softmax分数。
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])



#31. 如何找到numpy数组的百分位数？
# **难度等级：**L1
# **问题：**找到鸢尾属植物数据集的第5和第95百分位数
# 给定：
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
