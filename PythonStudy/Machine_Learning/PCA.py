from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits  import mplot3d

iris = load_iris()
x,y = iris.data,iris.target

pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(x)
print(pca3.explained_variance_ratio_)

pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(x)
print(pca2.explained_variance_ratio_)

ax = mplot3d.Axes3D(plt.figure(figsize=(4, 3)))
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=88, c=y, alpha=0.5)
plt.show()

plt.scatter(X2[:, 0], X2[:, 1], s=88, c=y, alpha=0.5)
plt.show()