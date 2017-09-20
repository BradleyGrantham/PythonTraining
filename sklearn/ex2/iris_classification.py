import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

'''
Classifying the iris dataset
'''

# load data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# split data into test and train
np.random.seed(0)
indices = np.random.permutation(len(iris_X))

iris_X_train = iris_X[indices[:-20]]
iris_y_train = iris_y[indices[:-20]]

iris_X_test = iris_X[indices[-20:]]
iris_y_test = iris_y[indices[-20:]]

logistic = linear_model.LogisticRegression(C=1e5)

print logistic.fit(iris_X_train, iris_y_train)

print logistic.predict(iris_X_test)
print iris_y_test
