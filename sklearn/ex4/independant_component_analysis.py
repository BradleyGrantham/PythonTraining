import numpy as np
import pandas as pd
from sklearn import decomposition
from scipy import signal

time = np.linspace(0, 10, 2000) # the 3rd argument is how many numbers you want between the first two arguments

s1 = np.sin(2 * time) # signal 1 is a sinusoidal signal
s2 = np.sign(np.sin(3*time)) # a square signal
s3 = signal.sawtooth(2 * np.pi * time) # saw tooth signal

S = np.c_[s1, s2, s3]

S += 0.2*np.random.normal(size=S.shape)
S /= S.std(axis=0) # very very neat way to standardize the data

A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])  # Mixing matrix

X = np.dot(S, A.T) # generate observations

# compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)

A_ = ica.mixing_.T
print np.allclose(X, np.dot(S_, A_) + ica.mean_)

'''
Most of this has gone completely over my head
'''