import numpy as np
import pandas as pd
import scipy as sp
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

'''
Connectivity constrained clustering
'''

try:  # SciPy >= 0.16 have face in misc
    from scipy.misc import face
    face = face(gray=True)
except ImportError:
    face = sp.face(gray=True)

# Resize it to 10% of the original size to speed up the processing
face = imresize(face, 0.10) / 255.

print face.shape # this will be the dimensions of the picture
X = np.reshape(face, (-1, 1))
print X.shape

connectivity = grid_to_graph(*face.shape)

print np.unique(connectivity.data)
