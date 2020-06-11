import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 数据加载
x = np.array(list(range(1, 11))).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).ravel()

# 创建模型
reg = DecisionTreeRegressor(max_depth=1)
reg2 = DecisionTreeRegressor(max_depth=3)
line_reg = linear_model.LinearRegression()
# 模型拟合
reg.fit(x,y)
reg2.fit(x,y)
line_reg.fit(x,y)
# 预测
x_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
y1 = reg.predict(x_test)
y2 = reg2.predict(x_test)
y3 = line_reg.predict(x_test)
# 图像打印
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_test, y1, color="cornflowerblue", label="max_depth=1", linewidth=2)
plt.plot(x_test, y2, color="yellowgreen", label="max_depth=3", linewidth=2)
plt.plot(x_test, y3, color='red', label='liner regression', linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
