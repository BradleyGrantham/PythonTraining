import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

iris_X_train = iris_X[:-20]
iris_y_train = iris_y[:-20]

iris_X_test = iris_X[-20:]
iris_y_test = iris_y[-20:]

'''
For many estimators it is important to first normalize the data
'''

svm1 = svm.SVC(kernel='linear')
print svm1.fit(iris_X_train, iris_y_train)

svm2 = svm.SVC(kernel='poly', degree=3)
print svm2.fit(iris_X_train, iris_y_train)

svm3 = svm.SVC(kernel='rbf')
print svm3.fit(iris_X_train, iris_y_train)

