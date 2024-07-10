import numpy as np


def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data