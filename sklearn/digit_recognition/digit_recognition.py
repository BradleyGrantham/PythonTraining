import numpy as np
from sklearn import datasets
from sklearn import linear_model, decomposition
from sklearn.model_selection import GridSearchCV


'''
Following on from the face recognition tutorial on sklearn, this script will hopefully
classify the digits in the digits dataset from sklearn using a LogisticRegression model
'''