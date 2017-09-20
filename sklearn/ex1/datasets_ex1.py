import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
print data.shape

digits = datasets.load_digits()
print digits.images.shape
print digits.images[-1].shape

data = digits.images.reshape((digits.images.shape[0], -1))
print data.shape

'''
Estimator objects take the data normally as a 2D array
Hence the data needs to be in the format of
(number of examples, number of features)

'''

