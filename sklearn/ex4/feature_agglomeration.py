import numpy as np
import pandas as pd
from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph
import matplotlib.pyplot as plt

digits = datasets.load_digits()
images = digits.images

X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                     n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)
X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)

'''
Agglo tranform reduces the dimensionality by finding similar features
 
So I believe this particular one will reduce the number of features to ~ 32
'''