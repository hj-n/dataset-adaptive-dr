import numpy as np
import autosklearn.regression
import os, json
from bayes_opt import BayesianOptimization
import umap
from zadu import zadu
import time


DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRIC = "tnc_25"
DR_TECHNIQUE = "umap"
pbounds = { "n_neighbors": (2, 200), "min_dist": (0.001, 0.999) }


technique_optimal_score_list = []
reducibility_list = {
	"mnc_25": [],
	"mnc_50": [],
	"mnc_75": [],
	"pds": []
}

spec = [{ "id": "tnc", "params": { "k": 25} }]
zadu_obj = zadu.ZADU(spec, data)

def extract_score(z_obj, embedding):
	score_dict = z_obj.measure(embedding, labels)[0]
	score_list = [score_dict[measure_name] for measure_name in measure_names]
	score = 2 * (score_list[0] * score_list[1]) / (score_list[0] + score_list[1]) if len(score_list) == 2 else score_list[0]
	return score if is_higher_better else -score

for dataset in DATASET_LIST:
	with open(f"./ground_truth/results/{DR_TECHNIQUE}_{DR_METRIC}.json") as f:
		ground_truth = json.load(f)
		technique_optimal_score_list.append(ground_truth[dataset]["score"])

	for reducibility in reducibility_list:
		with open(f"./exp/scores/reducibility/{reducibility}.json") as f:
			reducibility_data = json.load(f)
			reducibility_list[reducibility].append(reducibility_data[dataset]["score"])


if not os.path.exists("./app/results/predicted_tnc/9.json"):
	## run AutoML to find the predicted accuracy
	for trial in range(10):
		shuffled_indices = np.random.permutation(len(DATASET_LIST))
		shuffled_dataset_list = [DATASET_LIST[i] for i in shuffled_indices]
		shuffled_technique_optimal_score_list = [technique_optimal_score_list[i] for i in shuffled_indices]
		shuffled_reducibility_list = {
			"mnc_25": [reducibility_list["mnc_25"][i] for i in shuffled_indices],
			"mnc_50": [reducibility_list["mnc_50"][i] for i in shuffled_indices],
			"mnc_75": [reducibility_list["mnc_75"][i] for i in shuffled_indices],
			"pds": [reducibility_list["pds"][i] for i in shuffled_indices]
		}


		training_dataset = shuffled_dataset_list[:-16]
		testing_dataset = shuffled_dataset_list[-16:]

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
			per_run_time_limit=30,
			resampling_strategy='cv',
			memory_limit=10000,
			resampling_strategy_arguments={'folds': 5}
		)

		automl.fit(training_source, training_target)

		predictions = automl.predict(testing_source)

		result = {}

		for i in range(16):
			result[testing_dataset[i]] = {
				"prediction": float(predictions[i]),
				"true": float(testing_target[i])
			}

			with open(f"./app/results/predicted_tnc/{trial}.json", "w") as f:
				json.dump(result, f)


## run bayesian optimziation while using the predicted accuracy




