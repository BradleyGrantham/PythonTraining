import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import datasets, svm

digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

C_s = np.logspace(-6, -1, 10)

svc = svm.SVC(kernel='linear')

clf = GridSearchCV(estimator=svc, param_grid=dict(C=C_s), n_jobs=-1)
clf.fit(digits_X[:1000], digits_y[:1000])

print clf.best_score_
print clf.best_estimator_.C
print clf.score(digits_X[1000:], digits_y[1000:])

print cross_val_score(clf, digits_X, digits_y)