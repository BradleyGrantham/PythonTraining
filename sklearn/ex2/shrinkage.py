import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets

X = np.c_[0.5, 1].T
y = [0.5, 1]
test = np.c_[0,2].T
regr = linear_model.LinearRegression()

plt.figure()

np.random.seed(0)

for i in range(6):
    this_X = 0.1*np.random.normal(size=(2,1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

plt.savefig('fig1.png')

'''
Essentially what the above code is doing is adding a little
random bit onto X for each loop. It then fits a regression
line and plots it. I think it is just to prove that with only
two points, a small change in X means a big change in the 
regression line

High variance is the same as if you have a lot of data
and you fit a high degree polynomial. It fits the train data
very well but is unlikely to be the best solution for test data

High bias is underfitting(?). Doesn't fit the test data very well
Maybe a straight line isn't the best fit for the data (for example)
'''

# Now try ridge regression

ridge_regr = linear_model.Ridge(alpha=0.1)
plt.figure()

for i in range(6):
    this_X = 0.1*np.random.normal(size=(2,1)) + X
    ridge_regr.fit(this_X, y)
    plt.plot(test, ridge_regr.predict(test))
    plt.scatter(this_X, y)
plt.savefig('fig2.png')

'''
alpha can be chosen to minimize the left out error NOT ENTIRELY SURE WHAT THIS MEANS
use the diabetes dataset this time
'''

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

# np.random.seed(0)
# indices = np.random.permutation(len(diabetes_X))
indices = np.arange(len(diabetes_X))

diabetes_X_train = diabetes_X[indices[:-20]]
diabetes_X_test = diabetes_X[indices[-20:]]
diabetes_y_train = diabetes_y[indices[:-20]]
diabetes_y_test = diabetes_y[indices[-20:]]

alphas = np.logspace(-4, -1, 6) # return numbers in a logspace
# this particular line of code will return 6 numbers evenly space between
# 10^-4 and 10^-1

print diabetes_X_train.shape

print([ridge_regr.set_params(alpha=alpha
                             ).fit(diabetes_X_train, diabetes_y_train,
                                   ).score(diabetes_X_test, diabetes_y_test)
       for alpha in alphas])


'''
Ridge regression uses regularization. That is the (+LAMBDA) bit when using a cost
function (I think)

The chunk of code above is essentially showing that as we increase alpha
the R^2 value of our predictions is coming down
Hard to know whether this is what should happen given that I have no idea what the data
actually has in it
'''


