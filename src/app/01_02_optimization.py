import umap
from zadu import zadu
import time
import os, json
from bayes_opt import BayesianOptimization
import numpy as np
from tqdm import tqdm

## run optimization for 50 iterations (10 inits, 40 acq)

DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRIC = "tnc_25"
DR_TECHNIQUE = "umap"
pbounds = { "n_neighbors": (2, 200), "min_dist": (0.001, 0.999) }

DATA_POINT_MAXNUM = 3000

spec = [{ "id": "tnc", "params": { "k": 25} }]
measure_names = ["trustworthiness", "continuity"]

def extract_score(z_obj, embedding):
	score_dict = z_obj.measure(embedding)[0]
	score_list = [score_dict[measure_name] for measure_name in measure_names]
	score = 2 * (score_list[0] * score_list[1]) / (score_list[0] + score_list[1]) if len(score_list) == 2 else score_list[0]
	return score 

def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data

for idx in tqdm(range(10)):
	with open(f"./app/results/predicted_tnc/{idx}.json") as f:
		predicted_tnc = json.load(f)

	if not os.path.exists("./app/results/optimization/"):
		os.makedirs("./app/results/optimization/")

	## iterate through keys
	for dataset in predicted_tnc:
		if os.path.exists(f"./app/results/optimization/{idx}_{dataset}.json"):
			continue


		predicted_acc = predicted_tnc[dataset]["prediction"]
		if (predicted_acc > 0.99):
			predicted_acc = 0.99

		data = load_dataset(os.path.join(DATASET_PATH, dataset +  "/data.npy"))
		if data.shape[0] > DATA_POINT_MAXNUM:
			filterer= np.random.choice(data.shape[0], DATA_POINT_MAXNUM, replace=False)
			data = data[filterer]
		if DR_TECHNIQUE == "umap":
			zadu_obj = zadu.ZADU(spec, data)
			def run_method(n_neighbors, min_dist):
				try:
					n_neighbors = int(n_neighbors)
					min_dist = float(min_dist)
					embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(data)
					return extract_score(zadu_obj, embedding)
				except:
					return -1


		optimizer = BayesianOptimization(
			f=run_method,
			pbounds=pbounds,
			random_state=1,
			allow_duplicate_points=True
		)

		start = time.time()
		optimizer.maximize(
			init_points=10,
			n_iter=40
		)
		end = time.time()

		score = optimizer.max['target']
		time_used = end - start

		optimizer = BayesianOptimization(
			f=run_method,
			pbounds=pbounds,
			random_state=1,
			allow_duplicate_points=True
		)

		start = time.time()
		for i in range(50):  # Total of 50 iterations (10 initial points + 40 iterations)
			optimizer.maximize(
					init_points=1 if i < 10 else 0,  # 10 initial points
					n_iter=1 if i >= 10 else 0,  # 1 iteration at a time
			)
			if optimizer.max["target"] >= predicted_acc:
				break
		end = time.time()

		pred_score = optimizer.max['target']
		pred_time_used = end - start

		result = {
			"score": score,
			"time": time_used,
			"pred_score": pred_score,
			"pred_time": pred_time_used
		}



		with open(f"./app/results/optimization/{idx}_{dataset}.json", "w") as f:
			json.dump(result, f)

		## save the score to a file

