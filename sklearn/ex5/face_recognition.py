import numpy as np
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # randomly assigns a train and test set
from sklearn.model_selection import GridSearchCV # used this before, used for optimisation
from sklearn.datasets import fetch_lfw_people # this is simply to get the dataset
from sklearn.metrics import classification_report # think just what it says on the tin
from sklearn.metrics import confusion_matrix # a matrix to describe how 'good' the classification was
from sklearn import decomposition # Principal component analysis for reducing dimensionality
from sklearn import svm # support vector machine

# #############################################################################
# this loading data part has been taken straight from the sklearn webpage
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################

# split into a test set and a train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, # test_size is the percentage of data to be used for testing
                                                    random_state=0) # random state is simply the seed

# now we need to reduce the dimensionality because there are currently 1850 features

pca = decomposition.PCA()
pca.fit(X)
plt.figure()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.savefig('face_recognition_pca_fig.png')

# it is clear from this that we could easily reduce the dimensionality down to < 200
# choose 150 (because this is what sklearn have chosen on their website)

n_components = 150
pca = decomposition.PCA(n_components=n_components)
pca.fit(X)
eigenfaces = pca.components_.reshape((n_components, h, w))


# then apply the transform we have just fitted above to both our test and train sets
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

# now we want to fit a support vector machine to the data
# we are gonna choose an rbf model because once again this is what sklearn have chosen to use on their website

svc = svm.SVC(kernel='rbf') # pretty sure class weight balanced is default

# param_grid was copied from the sklearn web page. Essentially these are just the values we want
# to optimise when we use GridCV in a little while
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(svc, param_grid=param_grid)
clf.fit(X_train, y_train)

print "The best estimator found by the grid search was:"
print clf.best_estimator_

# now we have fit our model we can use it to predict

y_pred = clf.predict(X_test)
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))




