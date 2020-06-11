from sklearn import neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.RandomState(0)

# 数据加载
iris = datasets.load_iris()
# print(iris)
# 划分训练集与测试集
x,y = iris.data,iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
# print(x.shape,y.shape)
# print(x_train.shape,x_test.shape)

# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train,x_test)

# 创建模型
knn =neighbors.KNeighborsClassifier(n_neighbors=12)
# 模型拟合
knn.fit(x_train,y_train)
# 交叉验证
scores = cross_val_score(knn,x_train,y_train,cv=5,scoring='accuracy')
print(scores)
# 预测
y_pred = knn.predict(x_test)
# 评估
print(accuracy_score(y_test,y_pred))