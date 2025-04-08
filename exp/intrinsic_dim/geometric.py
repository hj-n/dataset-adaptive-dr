import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
from scipy.spatial.distance import cdist

## import linear regression
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler




def intrinsic_dim_geometric(data, _=None):

	dist_matrix = cdist(data, data)


	dist_matrix = dist_matrix / np.max(dist_matrix)

	curr_thrd = 0.000000001

	thrd_list = []
	ratio_list = []
	log_thrd_list = []
	log_ratio_list = []
	while True:
		## count number of points within distance curr_thrd
		count = np.sum((dist_matrix < curr_thrd).astype(int)) - data.shape[0]
		ratio = count / (data.shape[0] * (data.shape[0] - 1))


		thrd_list.append(curr_thrd)
		ratio_list.append(ratio)

		## set diagnoal to inf

		curr_thrd *= 2

		if ratio == 1:
			break

		if curr_thrd > 1:
			break
	
	
	filterer = np.array(ratio_list) > 0

	thrd_list = np.array(thrd_list)[filterer]
	ratio_list = np.array(ratio_list)[filterer]

	log_thrd_list = np.log(thrd_list)
	log_ratio_list = np.log(ratio_list)

	## get the slope of the log-log plot by linear regression
	log_thrd_list = np.array(log_thrd_list)
	log_ratio_list = np.array(log_ratio_list)

	log_thrd_list = log_thrd_list.reshape(-1, 1)
	log_ratio_list = log_ratio_list.reshape(-1, 1)


	reg = LinearRegression().fit(log_thrd_list, log_ratio_list)

	return reg.coef_[0][0]
	
	
	