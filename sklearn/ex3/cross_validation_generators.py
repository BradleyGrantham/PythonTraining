import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, svm

X = ["a", "a", "b", "c", "c", "c"]

k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print("Train: %s | Test: %s" % (train_indices, test_indices))

digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target
svc = svm.SVC(C=1, kernel='linear')

scores = [svc.fit(digits_X[train_indices], digits_y[train_indices]
         ).score(digits_X[test_indices], digits_y[test_indices]
                 ) for train_indices, test_indices in k_fold.split(digits_X)]

cv_scores = cross_val_score(svc, digits_X, digits_y, cv=k_fold, n_jobs=-1, scoring='precision_macro')
'''
n_jobs is to do with the CPUs that the computation is dist. over
n_jobs=-1 means it will be dist. over all of them
'''

print cv_scores