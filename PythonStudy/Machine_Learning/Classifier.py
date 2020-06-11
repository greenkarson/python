import numpy as np
from sklearn import datasets,linear_model,neighbors,svm,preprocessing
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix

# 数据加载
iris = datasets.load_iris()
x,y = iris.data,iris.target
# 划分训练集与测试集
x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.33)
# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 创建模型
clr = linear_model.SGDClassifier()
# clr = neighbors.KNeighborsClassifier()
# clr = linear_model.LogisticRegression()
# clr = svm.SVC()
# 模型拟合
clr.fit(x_train,y_train)
# 交叉验证
scores = cross_val_score(clr,x_train,y_train)
print(scores)
# 预测
y_pred = clr.predict(x_test)
# 评估
print(accuracy_score(y_test,y_pred))
# f1_score
print(f1_score(y_test, y_pred, average='micro'))
# 分类报告
print(classification_report(y_test, y_pred))
# 混淆矩阵
print(confusion_matrix(y_test, y_pred))