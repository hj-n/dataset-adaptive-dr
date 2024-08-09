import numpy as np

import os, json
from tqdm import tqdm
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

from scipy.stats import pearsonr

from scipy.stats import spearmanr


## turn off warnings

import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

SOURCES = [
	# "/intrinsic_dim/embedding_intdim",
	# "/intrinsic_dim/geometric_intdim",
	"/reducibility/mnc_25",
	# "/reducibility/mnc_50",
	"/reducibility/mnc_75",
	"/reducibility/mnc_100",
	# "/reducibility/pds",
]

source_names = {
	# "/intrinsic_dim/embedding_intdim": "projection_intdim",
	# "/intrinsic_dim/geometric_intdim": "geometric_intdim",
	"/reducibility/mnc_25": "mnc_25",
	# "/reducibility/mnc_50": "mnc_50",
	"/reducibility/mnc_75": "mnc_75",
	"/reducibility/mnc_100": "mnc_100",
}



def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data


## construct maximum accruacy achivable by the dataset for each DR quality metrics

### data storage
dr_metrics_list = []
competitors_list = []
regression_models_list = []
correlations_list = []


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

			print(f"- Target: {metric}")

			
			## Linear Regression with cross validation (100 times)
			r2_arr = []
			for _ in tqdm(range(100)):
				## shuffle the source/ target scores
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				reg = LinearRegression()
				r2s = cross_val_score(reg, source_shuffled, target_shuffled, cv=5, scoring='r2')
				r2_arr.append(r2s.mean())
			
			linear_max = np.max(r2_arr)
			
			## polynomical regression
			r2_arr = []
			for _ in tqdm(range(100)):
				poly = PolynomialFeatures(degree=2)
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				source_poly = poly.fit_transform(source_shuffled)
				reg = LinearRegression()
				r2s = cross_val_score(reg, source_poly, target_shuffled, cv=5, scoring='r2')
				r2_arr.append(r2s.mean())
			
			poly_max = np.max(r2_arr)

		
			## knn regression
			def knn_regressor(n_neighbors, weights):
				n_neighbors = int(n_neighbors)
				weights = weights

				weights = "uniform" if weights < 0.5 else "distance"

				knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

				r2s = cross_val_score(knn, source_shuffled, target_shuffled, cv=5, scoring='r2')
				return r2s.mean()
			
			pbounds = {'n_neighbors': (1, 20), 'weights': (0, 1)}
			r2_arr = []
			for _ in tqdm(range(100)):
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				optimizer = BayesianOptimization(
					f=knn_regressor,
					pbounds=pbounds,
					random_state=1,
					allow_duplicate_points=True,
					verbose=0
				)

				optimizer.maximize(init_points=5, n_iter=25)

				r2_arr.append(optimizer.max['target'])
			
			knn_max = np.max(r2_arr)

			## random forest regression
			def rf_regressor(n_estimators, max_depth, criterion):
				n_estimators = int(n_estimators)
				max_depth = int(max_depth)
				criterion = {
					0: "squared_error",
					1: "friedman_mse",
					2: "absolute_error",
					3: "poisson"
				}[int(np.floor(criterion))]

				rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

				r2s = cross_val_score(rf, source_shuffled, target_shuffled, cv=5, scoring='r2')
				return r2s.mean()
			
			pbounds = {'n_estimators': (5, 100), 'max_depth': (1, 10), 'criterion': (0, 3.99)}

			r2_arr = []
			for _ in tqdm(range(100)):
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				optimizer = BayesianOptimization(
					f=rf_regressor,
					pbounds=pbounds,
					random_state=1,
					allow_duplicate_points=True,
					verbose=0
				)

				optimizer.maximize(init_points=5, n_iter=25)

				r2_arr.append(optimizer.max['target'])
			
			rf_max = np.max(r2_arr)

			## gradient boosting regression
			def gb_regressor(n_estimators, max_depth, learning_rate):
				n_estimators = int(n_estimators)
				max_depth = int(max_depth)
				learning_rate = learning_rate

				gb = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

				r2s = cross_val_score(gb, source_shuffled, target_shuffled, cv=5, scoring='r2')
				return r2s.mean()

			pbounds = {'n_estimators': (5, 200), 'max_depth': (1, 10), 'learning_rate': (0.01, 0.5)}

			r2_arr = []
			for _ in tqdm(range(100)):
				shuffled_indices = np.random.permutation(len(source_scores))
				source_shuffled = source_scores[shuffled_indices]
				target_shuffled = target_scores[shuffled_indices]
				optimizer = BayesianOptimization(
					f=gb_regressor,
					pbounds=pbounds,
					random_state=1,
					allow_duplicate_points=True,
					verbose=0
				)

				optimizer.maximize(init_points=5, n_iter=25)

				r2_arr.append(optimizer.max['target'])
			
			gb_max = np.max(r2_arr)


			correlations_list += [linear_max, poly_max, knn_max, rf_max, gb_max]
			dr_metrics_list += [metric] * 5
			competitors_list += [source_names[source]] * 5
			regression_models_list += ["linear", "polynomial", "knn", "rf", "gb"]





df = pd.DataFrame({
	"dr_metric": dr_metrics_list,
	"competitor": competitors_list,
	"regression_model": regression_models_list,
	"r2": correlations_list
})



if not os.path.exists("./exp/results/"):
	os.makedirs("./exp/results/")

df.to_csv("./exp/results/correlations.csv", index=False)

			
			




