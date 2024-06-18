import numpy as np 
from ..helpers import generate_gaussian
from ...naive_mnc import mutual_neighbor_consistency
from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os

if os.path.exists('./src/exps/mnc_growth/data/mnc_growth_vs_d.csv' ):
	df = pd.read_csv('./src/exps/mnc_growth/data/mnc_growth_vs_d.csv')
	d_final_list = df['d'].values
	mnc_score_list = df['mnc_score'].values
	k_final_list = df['k'].values


else:

	mnc_score_list = []
	d_final_list = []
	k_final_list = []

	d_list = np.arange(2, 1002, 2)

	k_option_list = [5, 10, 15]
	n_sample = 3000

	for d in tqdm(d_list):
		for k in k_option_list:
			data = generate_gaussian(d, n_sample)
			mnc_score = mutual_neighbor_consistency(data, k)

			mnc_score_list.append(mnc_score)
			d_final_list.append(d)

			k_final_list.append(k)

	
	df = pd.DataFrame({
		'd': d_final_list,
		'mnc_score': mnc_score_list,
		'k': k_final_list
	})

	df.to_csv('./src/exps/mnc_growth/data/mnc_growth_vs_d.csv', index=False)


mnc_score_list = np.array(mnc_score_list)
d_final_list = np.array(d_final_list)
k_final_list = np.array(k_final_list)

# d_final_list = np.log(d_final_list)



## convert mnc to linear scale, assuming that the original follows the logistic function


# d_final_list = np.log(d_final_list)

# mnc_score_list = np.log(mnc_score_list)

sns.set(style="whitegrid")



plt.figure(figsize=(10, 6))

sns.scatterplot(
	x=d_final_list,
	y=mnc_score_list,
	hue=k_final_list,
	palette='tab10',
	alpha=0.2,
	linewidth=0,
	s=50
)

plt.savefig('./src/exps/mnc_growth/figures/mnc_growth_vs_d.png')



	