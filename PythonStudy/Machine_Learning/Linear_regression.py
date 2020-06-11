from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,explained_variance_score


# 数据加载
x,y =datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# 数据预处理

# 创建模型
reg = linear_model.LinearRegression()
# 岭回归
# reg = linear_model.Ridge(0.5)
# LASSO回归
# reg = linear_model.Lasso(0.1)
# 弹性回归
# reg = linear_model.ElasticNet(0.5,0.5)
# 逻辑斯蒂回归?
# reg = linear_model.LogisticRegression
# 贝叶斯回归
# reg = linear_model.BayesianRidge()
# 模型拟合
reg.fit(x_train,y_train)
# 交叉验证

# 预测
print(reg.coef_,reg.intercept_)
_x = np.array([-2.5,2.5])
_y = reg.coef_ * _x + reg.intercept_
y_pred =reg.predict(x_test)
# 评估
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
print(explained_variance_score(y_test,y_pred))
print()

# plt.scatter(x_test,y_test)
# plt.plot(_x,_y,linewidth =3,color = "orange")
# plt.show()