import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target
svc = svm.SVC(C=1, kernel='linear')

print svc.fit(digits_X[:-100], digits_y[:-100]
              ).score(digits_X[-100:], digits_y[-100:])

'''
We can also split the data into folds and test the scores
on each of the folds
'''

X_folds = np.split(digits_X, 3)
y_folds = np.split(digits_y, 3)
scores = list()

for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print scores

