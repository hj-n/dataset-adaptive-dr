from scipy.spatial.distance import cdist, pdist
import numpy as np
from knnsnn import DistMat as DistMat
from numba import cuda


def pairwise_distnace_shift(data):
	"""
	Pairwise distance shift: complexity metric targetting global structure
	
	INPUT:
	- data: numpy array of shape (n_samples, n_features) representing the high-dimensional data

	OUTPUT:
	- float: pairwise distance shift value


	"""
	dist_matrix = DistMat().distance_matrix(data)
	dist_matrix = dist_matrix.flatten()
	dist_matrix = dist_matrix[dist_matrix > 0]
	return np.std(dist_matrix) / np.mean(dist_matrix)
