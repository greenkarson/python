import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
# 数据加载
rng = np.random.RandomState(0)
X = 5 * rng.rand(100, 1)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
# 创建模型
kr = GridSearchCV(KernelRidge(),
                  param_grid={"kernel": ["rbf", "laplacian", "polynomial", "sigmoid"],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
# 模型拟合
kr.fit(X, y)

print(kr.best_score_, kr.best_params_)

X_plot = np.linspace(0, 5, 100)
y_kr = kr.predict(X_plot[:, None])

plt.scatter(X, y)
plt.plot(X_plot, y_kr, color="red")
plt.show()
