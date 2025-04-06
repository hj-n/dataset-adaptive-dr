import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import sys
from scipy.spatial.distance import cdist

## import linear regression

from sklearn.preprocessing import StandardScaler

import umap

def intrinsic_dim_projection(data, K):
	pca = PCA(n_components=min(data.shape[0], data.shape[1]))
	pca.fit(data)

	## number of dimensions needed to explain 95% of the variance
	exploained_variance = pca.explained_variance_ratio_
	cumulative_variance = np.cumsum(exploained_variance)
	dim = np.argmax(cumulative_variance >= 0.95) + 1

	dim = int(dim)

	return dim

