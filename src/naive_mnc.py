from .knnsnn import KnnSnn as ks
import numpy as np

def mutual_neighbor_consistency(data, k):
	"""
	Mutual Neighbor Consistency: complexity metric targetting local structure
	
	INPUT:
	- data: numpy array of shape (n_samples, n_features) representing the high-dimensional data
	- k: int, number of nearest neighbors to consider for computing mutual neighbor consistency

	OUTPUT:
	- float: mutual neighbor consistency value

	"""
	kSnn = ks(k)

	knn_indices = kSnn.knn(data)
	snn_matrix = kSnn.snn(knn_indices)

	## convert knn indices to knn distance matrix
	knn_matrix = np.zeros((data.shape[0], data.shape[0]))

	

	for i in range(data.shape[0]):
		for j in range(k):
			knn_matrix[i, knn_indices[i, j]] = 1
	
	# for i in range(data.shape[0]):
	# 	for j in range(data.shape[0]):
	# 		if snn_matrix[i, j] == 1:
	# 			print(i, j)
	# 			print(knn_indices[i, :])
	# 			print(knn_indices[j, :])
	# 			print()
	
	# print((k ** 2) / data.shape[0])
	# print((np.sum(snn_matrix)/2) /  (data.shape[0] * (data.shape[0] - 1) / 2))
	consistency = snn_matrix * knn_matrix

	# print(np.mean(consistency))

	return np.mean(consistency)
	

	

	
	