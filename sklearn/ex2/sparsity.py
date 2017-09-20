import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
The curse of dimensionality

As the dimensionality of the space increases, the volume of the space
increases exponentially. Therefore to properly 'fill' the space
you needd a lot more points than you did with one fewer dimensions

Ridge regression is good because it penalizes non informative features
However it doesn't set them to 0. Lasso is a better option

Lasso - least absolute shrinkage and selection operator

This is known as a sparse method (analogous to to Occams razor)
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

alphas = np.logspace(-4, -1, 6)

regr = linear_model.Lasso()

scores = [regr.set_params(alpha=alpha
                          ).fit(diabetes_X_train, diabetes_y_train
                                ).score(diabetes_X_test, diabetes_y_test)
          for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha

print regr.fit(diabetes_X_train, diabetes_y_train)
print regr.coef_
print regr.score(diabetes_X_test, diabetes_y_test)

'''
if you compare this score to the scores we got on the previous exercise (shrinkage)
this score is actually better than any we got on the previous page. I guess this means
the Lasso method is better for this particular dataset
'''


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(diabetes_X_test[:, 8], diabetes_X_test[:, 9], diabetes_y_test)
regr.fit(diabetes_X_train[:, [8, 9]], diabetes_y_train)
X3, X4 = np.meshgrid(diabetes_X_test[:, 8], diabetes_X_test[:, 9])
ax.plot_surface(X3, X4, regr.predict(diabetes_X_test[:, [8, 9]]), alpha = 0.05)
fig.savefig('3d_diabetes.png')

'''
The plot hasn't really worked. Looks like it has plotted a few surfaces rather than
just the one
'''

