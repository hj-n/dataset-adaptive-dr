import numpy as np 
from ..helpers import generate_gaussian
from ...pds import pairwise_distance_shift
from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os

## if file exists
if os.path.exists('./src/exps/pds_growth/data/pds_growth.csv') and False:
	df = pd.read_csv('./src/exps/pds_growth/data/pds_growth.csv')
	n_features_list = df['dimension'].values
	pds_score_list = df['pds_score'].values

else:

## make the features to have integer value between 1 and 10000 following log scale (bigger, more sparse)
	n_features_list = np.logspace(1, 3, num=500, base=10).astype(int)

	pds_score_list = []
	std_list = []


	for n_features in tqdm(n_features_list):
		n_samples = np.random.randint(100, 2000)
		## make random pdf using the combination of multiple random Gaussian distributions
		pdf = np.zeros(10000)
		Gaussian_n = np.random.randint(1, 100)
		for i in range(Gaussian_n):
			mean = np.random.randint(1, 10000)
			## make std range from 0 to 2000
			std = np.random.randint(1, 2000)
			height = np.random.randint(1, 1000)

			pdf += height * np.exp(-((np.arange(10000) - mean) ** 2) / (2 * std ** 2))
		
		## make the sum of pdf to be 1
		pdf = pdf / np.sum(pdf)


		## make data to follow a distribution where each dimension follows pdf (so every dimension follows same pdf)

		data = np.zeros((n_samples, n_features))
		for i in range(n_features):
			data[:, i] = np.random.choice(pdf, n_samples)


		
		

		pds_score = pairwise_distance_shift(data)
		pds_score_list.append(pds_score)


	df = pd.DataFrame({
		'dimension': n_features_list,
		'pds_score': pds_score_list
	})

	## save the data

	

	df.to_csv('./src/exps/pds_growth/data/pds_growth.csv', index=False)



## plot the results using regplot

n_features_list_log = np.log10(n_features_list)




sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.scatterplot(
	x=n_features_list_log,
	y=pds_score_list,
	s=100,
	alpha=0.1,
	color='black',
	## stroke width and color
	# linewidth=5

	
)


## save the plot

plt.savefig('./src/exps/pds_growth/figures/pds_growth.png', dpi=300)







