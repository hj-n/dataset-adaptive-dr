import os
import numpy as np
import hashlib


def load_file(dataset_path):
	data = np.load(dataset_path)
	return data

def load_names(dataset_path):
	dataset_names = os.listdir(dataset_path)
	return dataset_names

def load_dataset(
	dataset_path: str, ## path to the directory containing the datasets
	dataset_name, ## name of the dataset
	data_point_maxnum: int = 3000, ## maximum number of data points to load
	load_labels: bool = True ## whether to load labels or not
):

	## seed for reproducibility
	dataset_name_bytes = dataset_name.encode('utf-8')
	dataset_name_hash = hashlib.sha256(dataset_name_bytes).hexdigest()
	seed_int = int(dataset_name_hash, 16) % (2**32)
	np.random.seed(seed_int)


	data = load_file(os.path.join(dataset_path, dataset_name +  "/data.npy"))
	if load_labels:
		labels = load_file(os.path.join(dataset_path, dataset_name +  "/label.npy"))
	else:
		labels = None
	if data.shape[0] > data_point_maxnum:
		filterer= np.random.choice(data.shape[0], data_point_maxnum, replace=False)
		data = data[filterer]
		labels = labels[filterer]
	
	return data, labels
