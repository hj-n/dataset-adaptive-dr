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


# min_mnc = {
# 	5: np.min(mnc_score_list[k_final_list == 5]),
# 	10: np.min(mnc_score_list[k_final_list == 10]),
# 	15: np.min(mnc_score_list[k_final_list == 15])
# }


# mnc_truncated = []
# for i, mnc_score in enumerate(mnc_score_list):
# 	if k_final_list[i] == 5:
# 		mnc_truncated.append(mnc_score - min_mnc[5])
# 	elif k_final_list[i] == 10:
# 		mnc_truncated.append(mnc_score - min_mnc[10])
# 	else:
# 		mnc_truncated.append(mnc_score - min_mnc[15])


# mnc_truncated = np.array(mnc_truncated)





# mnc_truncated = mnc_truncated[k_final_list == 5]
# d_final_list = d_final_list[k_final_list == 5]
# k_final_list = k_final_list[k_final_list == 5]

# mnc_truncated = -np.log(mnc_truncated)
# d_final_list = np.log(d_final_list)


# # mnc_truncated = np.log(mnc_truncated)

# # d_final_list = np.log(d_final_list)

# plt.clf()

# plt.figure(figsize=(10, 6))

# sns.scatterplot(
# 	x=d_final_list,
# 	y=mnc_truncated,
# 	hue=k_final_list,
# 	palette='tab10',
# 	alpha=0.2,
# 	linewidth=0,
# 	s=50
# )

# plt.savefig('./src/exps/mnc_growth/figures/mnc_growth_trunc_vs_d.png')