import numpy as np
import autosklearn.regression
import os, json 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]


technique_optimal_score_list = {
	"umap": {},
	"tsne": {},
	"pca": {},
	"lle": {},
	"isomap": {},
	"umato": {}
}

for technique in DR_TECHNIQUES:
	for metric in DR_METRICS:
		with open(f"./ground_truth/results/{technique}_{metric}.json") as f:
			ground_truth = json.load(f)
			technique_optimal_score_list[technique][metric] = []
			for dataset in DATASET_LIST:
				technique_optimal_score_list[technique][metric].append(ground_truth[dataset]["score"])

reducibility_list = {
	"mnc_25": [],
	"mnc_50": [],
	"mnc_75": [],
	"pds": []
}

for dataset in DATASET_LIST:
	for reducibility in reducibility_list:
		with open(f"./exp/scores/reducibility/{reducibility}.json") as f:
			reducibility_data = json.load(f)
			reducibility_list[reducibility].append(reducibility_data[dataset]["score"])



for trial in range(20, 30):

	shuffled_indices = np.random.permutation(len(DATASET_LIST))
	shuffled_dataset_list = [DATASET_LIST[i] for i in shuffled_indices]

	training_dataset = shuffled_dataset_list[:-16]
	testing_dataset = shuffled_dataset_list[-16:]

	shuffled_reducibility_list = {
		"mnc_25": [reducibility_list["mnc_25"][i] for i in shuffled_indices],
		"mnc_50": [reducibility_list["mnc_50"][i] for i in shuffled_indices],
		"mnc_75": [reducibility_list["mnc_75"][i] for i in shuffled_indices],
		"pds": [reducibility_list["pds"][i] for i in shuffled_indices]
	}

	for technique in DR_TECHNIQUES:
		for metric in DR_METRICS:
			
			shuffled_technique_optimal_score_list = [technique_optimal_score_list[technique][metric][i] for i in shuffled_indices]

			if os.path.exists(f"./app_2/training/results/{trial}_{technique}_{metric}.json"):
				continue

			print(f"##### running trial {trial}... for {technique} and {metric}")

		
			training_target = shuffled_technique_optimal_score_list[:-16]
			testing_target = shuffled_technique_optimal_score_list[-16:]

			training_source = np.array([
				shuffled_reducibility_list["mnc_25"][:-16],
				shuffled_reducibility_list["mnc_50"][:-16],
				shuffled_reducibility_list["mnc_75"][:-16],
				shuffled_reducibility_list["pds"][:-16]
			]).T

			testing_source = np.array([
				shuffled_reducibility_list["mnc_25"][-16:],
				shuffled_reducibility_list["mnc_50"][-16:],
				shuffled_reducibility_list["mnc_75"][-16:],
				shuffled_reducibility_list["pds"][-16:]
			]).T

			training_target = np.array(training_target)
			testing_target = np.array(testing_target)
			



			automl = autosklearn.regression.AutoSklearnRegressor(
				time_left_for_this_task=600,
				per_run_time_limit=20,
				memory_limit=10000,
				metric=autosklearn.metrics.r2
			)


			automl.fit(training_source, training_target)

			predictions = automl.predict(testing_source)

			score = r2_score(testing_target, predictions)

			print(score)


			with open(f"./app_2/training/results/{trial}_{technique}_{metric}.json", "w") as f:
				json.dump({
					"dataset": testing_dataset,
					"predictions": list(predictions),
					"true": list(testing_target),
					"score": score
				}, f)
			