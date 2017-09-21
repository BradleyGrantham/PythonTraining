import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# load the digits dataset
digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

# load both the PCA estimator (for dimensionality reduction)
# and the logistic regression estimator

pca = decomposition.PCA()
logistic = linear_model.LogisticRegression()

# set up a pipeline
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# plot pca spectrum (just plotting the unexplained variance for each feature)
pca.fit(digits_X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# set the variables we will loop through
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

'''
Parameters of the Pipeline object can be set using __
'''

estimator = GridSearchCV(pipe, dict(pca__n_components=n_components,
                                    logistic__C=Cs))
estimator.fit(digits_X, digits_y)

'''
We fit this complex estimator object to the digits dataset
Then much like before when we used GridSearchCV to find the optimum value of alpha
it finds the optimum value of the number of parameters
We then (below) plot this line onto the graph to see where it is
'''

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')

plt.legend(prop=dict(size=12))
plt.savefig('pipeline_fig.png')