import numpy as np 
from ..helpers import generate_gaussian
from ...pds import pairwise_distance_shift
from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os

## if file exists
if os.path.exists('./src/exps/pds_growth/data/pds_growth.csv'):
	df = pd.read_csv('./src/exps/pds_growth/data/pds_growth.csv')
	n_features_list = df['dimension'].values
	pds_score_list = df['pds_score'].values

else:

## make the features to have integer value between 1 and 10000 following log scale (bigger, more sparse)
	n_features_list = np.logspace(0, 4, num=100, base=10).astype(int)

	pds_score_list = []
	std_list = []
	for n_features in tqdm(n_features_list):
		n_samples = np.random.randint(100, 2000)
		data = generate_gaussian(n_features, n_samples)
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
	s=200,
	alpha=0.15,
	color='black',
	## stroke width and color
	linewidth=5

	
)


## save the plot

plt.savefig('./src/exps/pds_growth/figures/pds_growth.png', dpi=300)







