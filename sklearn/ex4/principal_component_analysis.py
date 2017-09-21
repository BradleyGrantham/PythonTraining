import numpy as np
from sklearn import decomposition

# we first create a signal with one 'obsolete' feature

x1 = np.random.random(size=100)
x2 = np.random.random(size=100)
x3 = x1+x2

X = np.c_[x1, x2, x3]

pca = decomposition.PCA()
pca.fit(X)

print pca.explained_variance_

'''
We see from this that the third feature doesn't actually
explain any of the variance so we can reduce it by setting
pca.n_components to 2, and then i assume it takes the 2 features
that explain the most variance
'''

pca.n_components= 2
X_reduced = pca.fit_transform(X)
print X_reduced.shape