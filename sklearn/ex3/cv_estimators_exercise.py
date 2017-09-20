import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, KFold

'''
This is the solution to exercise 2 at
http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

Didn't quite understand what this exercise wanted so had a quick look
at the solution. The solution needs to first find the CV value of alpha
by using non-autonomous methods (GridCV) and then that can be checked
against the automatic value that LassoCV generates
'''

# load the diabetes data
diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data
diabetes_y = diabetes.target

train_size = 400 # i just set this by looking at the data (~ 90%)

# initialize a lasso estimator
lasso = linear_model.Lasso()

# initialize a list of alphas to test
alphas = np.logspace(-6, 4, 10)

# initialize a GridCV estimator
clf = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas), cv=3, n_jobs=-1)

clf.fit(diabetes_X[:train_size], diabetes_y[:train_size])

print 'Alpha: ', clf.best_estimator_.alpha
print 'Best score: ', clf.best_score_
print 'Manual score: '. clf.score(diabetes_X[train_size:], diabetes_y[train_size:])
# print clf.cv_results_['mean_test_score']
# print clf.cv_results_['std_test_score']

'''
Now to check the alpha obtained in this
'manual' version compared to the automatic

I've copied this bit below for now
'''



lasso = linear_model.LassoCV()
k_fold = KFold(3)
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso.alpha_, lasso.score(X[test], y[test])))

