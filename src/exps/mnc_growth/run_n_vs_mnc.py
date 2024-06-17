import numpy as np
from ..helpers import generate_gaussian
from ...mnc import mutual_neighbor_consistency
from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os

## if file exists
if os.path.exists('./src/exps/mnc_growth/data/mnc_growth_vs_logn.csv' ) and False:
	df = pd.read_csv('./src/exps/mnc_growth/data/mnc_growth_vs_logn.csv')
	n_samples_final_list = df['sample_num'].values
	mnc_score_list = df['mnc_score'].values
	k_option_final_list = df['k_option'].values




else:
	mnc_score_list = []
	n_samples_final_list = []
	k_option_final_list = []

	# n_sample_list = np.arange(1000, 10000, 200)
	n_sample_list = np.logspace(3.48, 4.48, num=60, base=10).astype(int)
	k_option = ["fixed", "n^(2/3)", "n"]

	for n_sample in tqdm(n_sample_list):
		n_feature = 1000
		data = generate_gaussian(n_feature, n_sample)

		for k_strat in k_option:
			if k_strat == "fixed":
				k = 10
			elif k_strat == "n^(2/3)":
				k = int(n_sample ** (2/3)) / 20
			elif k_strat == "n":
				k = n_sample / 200
			
			k = int(k)
			
			mnc_score = mutual_neighbor_consistency(data, k)
			mnc_score_list.append(mnc_score)
			n_samples_final_list.append(n_sample)
			k_option_final_list.append(k_strat)


	df = pd.DataFrame({
		'sample_num': n_samples_final_list,
		'mnc_score': mnc_score_list,
		'k_option': k_option_final_list
	})

	## save the data

	df.to_csv('./src/exps/mnc_growth/data/mnc_growth_vs_logn.csv', index=False)



mnc_score_list = np.array(mnc_score_list)
n_samples_final_list = np.array(n_samples_final_list)
k_option_final_list = np.array(k_option_final_list)

mnc_score_list  = - np.log10(mnc_score_list)

n_samples_final_list  = np.log10(n_samples_final_list)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.scatterplot(
	x=n_samples_final_list,
	y=mnc_score_list,
	hue=k_option_final_list,
	palette="tab10",

	s=100,
	alpha=0.4,
	## stroke width and color
)

## save the plot

plt.savefig('./src/exps/mnc_growth/figures/mnc_growth_vs_logn.png', dpi=300)





