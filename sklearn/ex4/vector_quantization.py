import numpy as np
import pandas as pd
import scipy as sp
from sklearn import cluster

'''
I think what this code is essentially doing is compressing an image
down to only 5 colours (5 clusters in the kmeans)
It then assigns each old value to its new 'clustered' value

This is exactly what image compression is, reducing the number
of different colours
Although not entirely sure that this is what image compression is
to be honest
'''

try:
    face =sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)
X = face.reshape((-1, 1)) # this converts data to (n_sample, n_feature)
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

print face[0:10]
print np.unique(face_compressed)
