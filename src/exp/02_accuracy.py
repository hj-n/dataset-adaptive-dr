import numpy as np

import os, json
from tqdm import tqdm


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from scipy.stats import pearsonr

from scipy.stats import spearmanr


DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

SOURCES = [
	"/intrinsic_dim/embedding_intdim",
	"/intrinsic_dim/geometric_intdim",
	"/reducibility/mnc_naive_25",
	"/reducibility/mnc_25",
	"/reducibility/mnc_naive_50",
	"/reducibility/mnc_50",
	"/reducibility/mnc_naive_75",
	"/reducibility/mnc_75",
	"/reducibility/pds",
]



def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data


## construct maximum accruacy achivable by the dataset for each DR quality metrics
maximum_accuracy = {}
for metric in DR_METRICS:
	maximum_accuracy[metric] = []


for dataset in tqdm(DATASET_LIST):
	for metric in DR_METRICS:
		curr_accuracies = []
		for technique in DR_TECHNIQUES:
			with open(f"./ground_truth/results/{technique}_{metric}.json") as f:
				ground_truth = json.load(f)
				curr_accuracies.append(ground_truth[dataset]["score"])
		
		maximum_accuracy[metric].append(max(curr_accuracies))


for source in SOURCES:
	with open(f"./exp/scores{source}.json") as f:
		source_data= json.load(f)
		source_scores = []
		for dataset in DATASET_LIST:
			source_scores.append(source_data[dataset]["score"])

		# print(source_scores)
		source_scores = np.array(source_scores)
		source_scores = source_scores.reshape(-1, 1)
		print(f"Source: {source}")
		for metric in DR_METRICS:
			target_scores = maximum_accuracy[metric]
			target_scores = np.array(target_scores)
			target_scores = target_scores.reshape(-1, 1)

			print("- DR Metric:", end=" ")

			
			## Linear Regression with cross validation (100 times)
			r2_arr = []
			for _ in range(100):
				## shuffle the source/ target scores
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				reg = LinearRegression()
				r2s = cross_val_score(reg, source_shuffled, target_shuffled, cv=5, scoring='r2')
				r2_arr.append(r2s.mean())
			
			print("LR - ", np.max(r2_arr))


			
			




