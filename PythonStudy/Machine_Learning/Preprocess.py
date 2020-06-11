from sklearn import preprocessing
import numpy as np
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -3.]])

# 标准化
# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
# x_scaler = preprocessing.scale(x)
# print(x_scaler)
# print(x_scaler.mean(axis=0), x_scaler.std(0))

# 标准化
# scaler = preprocessing.StandardScaler()
# x_scale = scaler.fit_transform(x)
# print(x_scale)
# print(x_scale.mean(0), x_scale.std(0))

#minmax
# x_scaler = preprocessing.minmax_scale(x)
# print(x_scaler)
# print(x_scaler.mean(axis=0), x_scaler.std(0))

# MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler()
# x_scale = scaler.fit_transform(x)
# print(x_scale)
# print(x_scale.mean(0), x_scale.std(0))

# RobustScaler
# scaler = preprocessing.RobustScaler()
# x_scale = scaler.fit_transform(x)
# print(x_scale)
# print(x_scale.mean(0), x_scale.std(0))

# Normalizer
# scaler = preprocessing.Normalizer(norm="l2")
# x_scale = scaler.fit_transform(x)
# print(x_scale)
# print(x_scale.mean(0), x_scale.std(0))

# 二值化
# scaler = preprocessing.Binarizer(threshold=0)
# x_scale = scaler.fit_transform(x)
# print(x_scale)


# one_hot
# enc = preprocessing.OneHotEncoder(n_values=3, sparse=False)
# ans = enc.fit_transform([[0], [1], [2],[1]])
# print(ans)

# 缺失数据
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
# y_imp = imp.fit_transform([[np.nan, 2], [6, np.nan], [7, 6]])
# print(y_imp)
#
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
y_imp = imp.transform([[np.nan, 2], [6, np.nan], [7, 6]])
print(y_imp)
