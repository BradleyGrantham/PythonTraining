import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

'''
Some estimators have their own cross validation capability
Such as the LassoCV
This means that they split their data etc to find the best
possbile parameters based on training and test sets
'''

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

lasso = linear_model.LassoCV()
lasso.fit(diabetes_X, diabetes_y)

print lasso.alpha_

