import numpy as np 
from ..helpers import generate_gaussian
from ...mnc import mutual_neighbor_consistency
from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os

## if file exists
if os.path.exists('./src/exps/mnc_growth/data/mnc_growth.csv' ) and False:
	df = pd.read_csv('./src/exps/mnc_growth/data/mnc_growth.csv')
	n_samples_final_list = df['sample_num'].values
	mnc_score_list = df['mnc_score'].values
	cons_final_list = df['cons'].values

else:

## make the features to have integer value between 1 and 10000 following log scale (bigger, more sparse)
	# n_samples_list = [200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
	# k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70]
	# k_list = [2, 3, 4, 5]
	# n_sample_list = 
	mnc_score_list = []
	n_samples_final_list = []
	cons_final_list = []

	n_sample_list = [1000, 2000, 3000, 4000, 5000, 6000] 
	
	constant_list = [1000, 10000, 100000]
	for n_samples in tqdm(n_sample_list):
		n_feature = 1000
		for cons in tqdm(constant_list):
			# n_samples = int((k * (cons ** (1/3))) ** (3/2))
			k = int((n_samples ** (2/3)) / cons ** (1/3))
			if k < 1:
				continue
			for i in range(500):
				data = generate_gaussian(n_feature, n_samples)
				# mnc_score = mutual_neighbor_consistency(data, int(n_samples ** (2/3) / (1000 ** (1/3))))
				mnc_score = mutual_neighbor_consistency(data, k)
				mnc_score_list.append(mnc_score)
				n_samples_final_list.append(n_samples)
				cons_final_list.append(cons)



	df = pd.DataFrame({
		'sample_num': n_samples_final_list,
		'mnc_score': mnc_score_list,
		'cons': cons_final_list
	})

	## save the data

	df.to_csv('./src/exps/mnc_growth/data/mnc_growth.csv', index=False)

# print(np.mean(mnc_score_list))
# print(np.median(mnc_score_list))

## plot the results using regplot


mnc_score_list = np.array(mnc_score_list)
cons_final_list = np.array(cons_final_list)
n_samples_final_list = np.array(n_samples_final_list)

mnc_score_list = np.log10(mnc_score_list)
n_samples_final_list = n_samples_final_list
# n_features_final_list

# mnc_score_list = np.array(mnc_score_list)[n_features_final_list]
# n_features_final_list = np.array(n_features_final_list)[n_features_final_list]



## (\sqrt{3}/2)^x 
# n_features_final_list = np.log(n_features_final_list)


# n_features_single_list = []
# mnc_score_single_list = []
# for i in list(set(n_features_final_list)):
# 	sumsum = []
# 	count = 0
# 	for j in range(len(n_features_final_list)):
# 		if i - 50<n_features_final_list[j] < i+50:
# 			sumsum.append(mnc_score_list[j])
# 			count += 1
	
# 	avg = np.mean(sumsum)

# 	n_features_single_list.append(i)
# 	mnc_score_single_list.append(avg)



# n_features_single_list = np.array(n_features_single_list)
# mnc_score_single_list = np.array(mnc_score_single_list)

# n_features_single_list = np.log(n_features_single_list)

# filterer = n_features_single_list > 6000


# n_features_single_list = np.array(n_features_single_list)[filterer]
# mnc_score_single_list = np.array(mnc_score_single_list)[filterer]



# mnc_score_list = mnc_score_list[1000:]
# n_features_final_list = n_features_final_list[1000:]


# mnc_score_list = mnc_score_list[20:]
# n_features_list = n_features_final_list[20:]



sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.scatterplot(
	x=n_samples_final_list,
	y=mnc_score_list,
	hue=cons_final_list,
	s=40,
	alpha=0.1,
	palette='tab10'
	## stroke width and color
	# linewidth=5
)

## remove legent
# plt.legend([],[], frameon=False)


## save the plot

plt.savefig('./src/exps/mnc_growth/figures/mnc_growth.png', dpi=300)







