import matplotlib.pyplot as plt
from sklearn.datasets import samples_generator
from sklearn import metrics,cluster
from sklearn.mixture import gaussian_mixture

# x,y = samples_generator.make_blobs(n_samples=200,n_features=3,cluster_std=0.6,random_state=0)
x,y = samples_generator.make_circles(n_samples=200,noise=.05,random_state=0,factor=0.4)
# x,y = samples_generator.make_moons(n_samples=200,noise=.05,random_state=0)
# print(x.shape,y.shape)

# clu = cluster.KMeans(2)
# clu = cluster.MeanShift()
# clu = cluster.DBSCAN(eps=0.98,min_samples=4)
# clu = cluster.SpectralClustering(2,affinity="nearest_neighbors")
# clu = cluster.AffinityPropagation()
clu = gaussian_mixture.GaussianMixture(n_components=2)

labels = clu.fit_predict(x)
print(metrics.silhouette_score(x,labels))
print(metrics.calinski_harabasz_score(x,labels))
print(metrics.davies_bouldin_score(x,labels))
plt.scatter(x[:,0],x[:,1],c=labels)
plt.show()