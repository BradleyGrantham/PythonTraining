import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

# np.random.seed(0)
# indices = np.random.permutation(len(diabetes_X))
indices = np.arange(len(diabetes_X))

diabetes_X_train = diabetes_X[indices[:-20]]
diabetes_X_test = diabetes_X[indices[-20:]]
diabetes_y_train = diabetes_y[indices[:-20]]
diabetes_y_test = diabetes_y[indices[-20:]]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

print regr.coef_

predictions = regr.predict(diabetes_X_test)

mse = np.mean(np.power(np.subtract(predictions, diabetes_y_test), 2))

print mse

print regr.score(diabetes_X_test, diabetes_y_test)