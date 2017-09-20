import numpy as np
import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

'''
This will be a solution to the first exercise at
https://github.com/BradleyGrantham/PythonTraining/blob/master/sklearn/ex2/svm_exercise.py
'''

digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)
print C_s
k_fold = KFold(n_splits=3)

maxes, mins, means = list(), list(), list()

for C in C_s:
    svc.set_params(C=C)
    cv_scores = cross_val_score(svc, digits_X,
                                digits_y, cv=k_fold,
                                n_jobs=-1)
    maxes.append(max(cv_scores))
    mins.append(min(cv_scores))
    means.append(np.mean(cv_scores))

plt.figure()
plt.plot(C_s, maxes, linestyle='dashed')
plt.plot(C_s, mins, linestyle='dashed')
plt.plot(C_s, means)
plt.xscale('log')
plt.savefig('cross_val_digits_fig.png')
