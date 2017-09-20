import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_desicion_boundary(X, svm):
    '''

    :param X: the X data of the dataset
    :param svm: a support vector machine estimate
    :return: two arrays to be plotted (like an x and y coordinate)
    '''
    # plot the observations
    # first create a 'mesh grid'. From what I understand it is essentially a space over
    # which will have have lots of values of both X1 and X2
    x1_min, x1_max = iris_X[:, 0].min() - 1, iris_X[:, 0].max() + 1
    x2_min, x2_max = iris_X[:, 1].min() - 1, iris_X[:, 1].max() + 1
    X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                         np.arange(x2_min, x2_max, 0.01))

    y = svm.decision_function(np.c_[X1.ravel(), X2.ravel()])

    # returns a flattened array so we can predict only the values of X1, X2 where the decision
    # function is equal to 0
    X1 = X1.ravel()
    X2 = X2.ravel()
    y = np.round(y, 2)
    X1 = X1[y == 0.00]
    X2 = X2[y == 0.00]

    return X1, X2


#load the data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print np.unique(iris_y) # see what classes there are

# we only want classes 1 and 2 so get rid of the 0's
iris_X = iris_X[iris_y!=0, :2] # also only want first two features
iris_y = iris_y[iris_y!=0]

# randomly assign 10% to a test set
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
test_size = int(round(0.1*len(iris_X))) # testing size should be ~ 10% of data

iris_X_train = iris_X[indices[:-test_size]]
iris_y_train = iris_y[indices[:-test_size]]
iris_X_test = iris_X[indices[-test_size:]]
iris_y_test = iris_y[indices[-test_size:]]

# fit linear support vector machine to data
svm_lin = svm.SVC(kernel='linear')
svm_lin.fit(iris_X_train, iris_y_train)

svm_poly = svm.SVC(kernel='poly', degree=3)
svm_poly.fit(iris_X_train, iris_y_train)

svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(iris_X_train, iris_y_train)

# plot all of the findings
X1_lin, X2_lin = plot_desicion_boundary(iris_X_train, svm_lin)
X1_poly, X2_poly = plot_desicion_boundary(iris_X_train, svm_poly)
X1_rbf, X2_rbf = plot_desicion_boundary(iris_X_train, svm_rbf)


# finally ploy both the decision boundary and the points with the colours corresponding
# to their class
plt.figure()
plt.plot(X1_lin, X2_lin, c='blue')
plt.plot(X1_poly, X2_poly, c='red')
plt.plot(X1_rbf, X2_rbf, c='green')
plt.scatter(iris_X_train[:, 0], iris_X_train[:, 1], c=iris_y_train)
plt.savefig('iris_svm.png')