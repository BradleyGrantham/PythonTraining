import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model, decomposition
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix


'''
Following on from the face recognition tutorial on sklearn, this script will hopefully
classify the digits in the digits dataset from sklearn using a LogisticRegression model
'''

################################################################################
# firstly load the dataset in
digits = datasets.load_digits()
X = digits.data
y = digits.target

# print np.unique(y) # check to see that how theyre labelled (0-9 as expected)
print X.shape, y.shape  # print the shape of both X and y

# split the dataset into train and test (test is 20% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
################################################################################
# now to reduce the dimensionality of the problem
# start by plotting the explained variance against the number of features
# to see how many we are going to need
pca = decomposition.PCA()
pca.fit(X)

plt.figure()
plt.plot(np.arange(0, X.shape[1], 1), pca.explained_variance_)
plt.savefig('pca_explained_variance.png')

# we can actually use GridCV to find the optimum number of
# features as well before we even fit a model (I THINK)
# in this example the variance doesnt increase that much so we
# may as well keep the number of features at 64

logistic = linear_model.LogisticRegression()

C_s = np.logspace(0, 2, 4)

clf = GridSearchCV(logistic, param_grid=dict(C=C_s), n_jobs=-1)

clf.fit(X_train, y_train)

print clf.best_estimator_

y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

con_matr = confusion_matrix(y_test, y_pred)

print score
print con_matr
