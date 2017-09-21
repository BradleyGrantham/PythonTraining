import numpy as np
import pandas as pd
from sklearn import decomposition
from scipy import signal
import matplotlib.pyplot as plt

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

# all close returns true if two arrays are the same elementwise to a specified tolerance
print np.allclose(X, np.dot(S_, A_) + ica.mean_)


plt.figure()
plt.plot(time, s1, c='blue')
plt.plot(time, s2, c='green')
plt.plot(time, s3, c='red')
plt.savefig('original_signals.png')


'''
Most of this has gone completely over my head
'''