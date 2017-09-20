import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''
Solution to the exercise on 
http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#curse-of-dimensionality
'''

# load the dataset and split into X and y
digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

# print digits_X.shape, digits_y.shape

# split into a training set and test set by random indexing
np.random.seed(0)
indices = np.random.permutation(len(digits_X))
train_size = int(round(0.1*len(digits_X))) # training size should be ~ 10% of data

digits_X_train = digits_X[indices[:-train_size]]
digits_y_train = digits_y[indices[:-train_size]]

digits_X_test = digits_X[indices[-train_size:]]
digits_y_test = digits_y[indices[-train_size:]]

# apply a nearest neighbour approach to the data
knn = KNeighborsClassifier()
knn.fit(digits_X_train, digits_y_train)
# print knn.predict(digits_X_test)
# print digits_y_test
knn_score = knn.score(digits_X_test, digits_y_test)
knn_accuracy_score = accuracy_score(digits_y_test, knn.predict(digits_X_test))

# apply a logistic regression model to the data
lm = linear_model.LogisticRegression()
lm.fit(digits_X_train, digits_y_train)
# print lm.predict(digits_X_test)
# print digits_y_test
lm_score = lm.score(digits_X_test, digits_y_test)
lm_accuracy_score = accuracy_score(digits_y_test, lm.predict(digits_X_test))

print 'Linear model score: ' + str(round(lm_score, 2))
print 'Linear model accuracy score: ' + str(round(lm_accuracy_score, 2))
print 'KNN score: ' + str(round(knn_score, 2))
print 'KNN accuracy score: ' + str(round(knn_accuracy_score, 2))

# use shrinkage on logistic regression to see if it helps
lm_shrink = linear_model.LogisticRegression(penalty='l1')
lm_shrink.fit(digits_X_train, digits_y_train)
# print lm.predict(digits_X_test)
# print digits_y_test
lm_shrink_score = lm_shrink.score(digits_X_test, digits_y_test)
lm_shrink_accuracy_score = accuracy_score(digits_y_test, lm_shrink.predict(digits_X_test))

if lm_shrink_score > lm_score:
    print 'Linear model (shrinkage) score: ' + str(round(lm_shrink_score, 2))
    print 'Linear model (shrinkage) accuracy score: ' + str(round(lm_shrink_accuracy_score, 2))

# use sparsity on logistic regression to see if it helps
lm_spars = linear_model.LogisticRegression(penalty='l2')
lm_spars.fit(digits_X_train, digits_y_train)
# print lm.predict(digits_X_test)
# print digits_y_test
lm_spars_score = lm_spars.score(digits_X_test, digits_y_test)
lm_spars_accuracy_score = accuracy_score(digits_y_test, lm_spars.predict(digits_X_test))

if lm_spars_score > lm_score:
    print 'Linear model (sparsity) score: ' + str(round(lm_spars_score, 2))
    print 'Linear model (sparsity) accuracy score: ' + str(round(lm_spars_accuracy_score, 2))