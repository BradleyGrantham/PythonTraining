import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

# print iris_X.shape, iris_Y.shape

np.random.seed(0) # set the seed for pseduorandm generator
indices = np.random.permutation(len(iris_X))
print len(iris_X)
iris_X_train = iris_X[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_Y_test = iris_Y[indices[-10:]]
iris_Y_train = iris_Y[indices[:-10]]

print iris_X_train.shape, iris_Y_train.shape

nbrs = KNeighborsClassifier() # im going to leave all as default

nbrs.fit(iris_X_train, iris_Y_train)

print nbrs.predict(iris_X_test)
print iris_Y_test
