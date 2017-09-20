import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

X = np.c_[0.5, 1].T
y = [0.5, 1]
test = np.c_[0,2].T
regr = linear_model.LinearRegression()

plt.figure()

np.random.seed(0)

for i in range(6):
    this_X = .1*np.random.normal(size=(2,1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

plt.savefig('fig1.png')

'''
Essentially what the above code is doing is adding a little
random bit onto X for each loop. It then fits a regression
line and plots it. I think it is just to prove that with only
two points, a small change in X means a big change in the 
regression line
'''