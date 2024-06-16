
import numpy as np

## generate random standard Gaussian distribution with given dimensionality and point number
def generate_gaussian(dim, n_samples, std=1.0):
	return np.random.randn(n_samples, dim) * std 
