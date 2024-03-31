from knnsnn import KnnSnn as ks
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
	snn_results = kSnn.snn(knn_indices)

	## convert knn indices to knn distance matrix
	knn_distances = np.zeros((data.shape[0], data.shape[0]))

	for i in range(data.shape[0]):
		for j in range(k):
			knn_distances[i, knn_indices[i, j]] = (k - j) / k
	
	neighbor_consistency_sum = 0
	for i in range(data.shape[0]):
		knn_sim = knn_distances[i, :]
		snn_sim = snn_results[i, :]

		## compute neighbor consistency as cosine similarity
		cos_sim = np.dot(knn_sim, snn_sim) / (np.linalg.norm(knn_sim) * np.linalg.norm(snn_sim))
		neighbor_consistency_sum += cos_sim
	
	neighbor_consistency = 1 - neighbor_consistency_sum / data.shape[0]

	return neighbor_consistency
	

	

	
	