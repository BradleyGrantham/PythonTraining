import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(iris_X)

print iris_X.shape

print k_means.labels_[::10]
print iris_y[::10]

'''
Choosing the right number of clusters is hard
The algorithm is prone to falling into local minima
(Although sklearn reckon they've mitigated this risk)
'''

# plot the labels against two of the features just to see what it looks like
plt.figure()
plt.scatter(iris_X[:,0], iris_X[:,1], c=k_means.labels_)
plt.savefig('kmean_fig.png')