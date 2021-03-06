import numpy as np
from sklearn import datasets,preprocessing,ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据加载
wine = datasets.load_wine()
x,y = wine.data,wine.target

# 划分训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3 , random_state=0)
# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 创建模型
rf = ensemble.RandomForestClassifier(n_estimators=25,max_depth=3)
# 模型拟合
rf.fit(x_train,y_train)
# 交叉验证
# 预测
y_pred = rf.predict(x_test)
# 评估
print(accuracy_score(y_test,y_pred))