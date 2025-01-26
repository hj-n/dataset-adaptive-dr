import numpy as np 
import os, json
import ground_truth.helpers as hp
from zadu import zadu
from bayes_opt import BayesianOptimization
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
from sklearn.manifold import Isomap, LocallyLinearEmbedding

import umato


DR_METRIC = "tnc_25"
is_higher_better = True

DATASET_PATH = "../labeled-datasets/npy/"
measure_names = ["trustworthiness", "continuity"]

DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

DATA_POINT_MAXNUM = 2000

spec = [{ "id": "tnc", "params": { "k": 25} }]

def extract_score(z_obj, embedding):
	score_dict = z_obj.measure(embedding)[0]
	score_list = [score_dict[measure_name] for measure_name in measure_names]
	score = 2 * (score_list[0] * score_list[1]) / (score_list[0] + score_list[1]) if len(score_list) == 2 else score_list[0]
	return score 


def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data


for trial_idx in tqdm(range(10, 20)):
	with open(f"./app_2/training/results/{trial_idx}_umap_{DR_METRIC}.json") as f:
		data_file = json.load(f)
	
	datasets = data_file["dataset"]

	for dataset_idx, dataset in enumerate(datasets):

		# if os.path.exists(f"./app_2/pipeline/{trial_idx}_{dataset_idx}_dataset_adaptive.json"):
		# 	continue

		data = load_dataset(f"{DATASET_PATH}/{dataset}/data.npy")
		labels = load_dataset(f"{DATASET_PATH}/{dataset}/label.npy")
		if data.shape[0] > DATA_POINT_MAXNUM:
			filterer = np.random.choice(data.shape[0], DATA_POINT_MAXNUM, replace=False)
			data = data[filterer]
			labels = labels[filterer]
		
		zadu_obj = zadu.ZADU(spec, data)
		scores = []
		for technique in DR_TECHNIQUES:
			with open(f"./app_2/training/results/{trial_idx}_{technique}_{DR_METRIC}.json") as f:
				data_file = json.load(f)
				score = data_file["predictions"][dataset_idx]
				scores.append(score)
		
		## find the top-3 best technique and top-1 best technique
		top_3_idx = np.argsort(scores)[-3:]
		top_1_idx = np.argsort(scores)[-1:]
		top_3_techniques = [DR_TECHNIQUES[idx] for idx in top_3_idx]
		top_1_techniques = [DR_TECHNIQUES[idx] for idx in top_1_idx][0]

		top_3_times = 0
		top_1_times = 0
		top_3_scores = -1
		top_1_scores = -1

		def extract_score(z_obj, embedding):
			score_dict = z_obj.measure(embedding, labels)[0]
			score_list = [score_dict[measure_name] for measure_name in measure_names]
			score = 2 * (score_list[0] * score_list[1]) / (score_list[0] + score_list[1]) if len(score_list) == 2 else score_list[0]
			return score if is_higher_better else -score
	


		for idx in range(3):
			
			method = top_3_techniques[idx]

			if method == "pca":
				def run_method():
					embedding = PCA(n_components=2).fit_transform(data)
					return extract_score(zadu_obj, embedding)

			if method == "umap":
				def run_method(n_neighbors, min_dist):
					embedding = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist)).fit_transform(data)
					return extract_score(zadu_obj, embedding)
				pbounds = {'n_neighbors': (2, 200), 'min_dist': (0.001, 0.999)}
			elif method == "tsne":
				def run_method(perplexity):
					perplexity = int(perplexity)
					if perplexity >= data.shape[0]:
						perplexity = data.shape[0] - 1
					embedding = TSNE(perplexity=perplexity).fit_transform(data)
					return extract_score(zadu_obj, embedding)
				pbounds = {'perplexity': (2, 500)}
			elif method == "umato":
				def run_method(n_neighbors, min_dist, hub_num):
					try:
						if hub_num >= data.shape[0]:
							hub_num = data.shape[0] - 1
						if n_neighbors >= data.shape[0]:
							n_neighbors = data.shape[0] - 1
						embedding = umato.UMATO(n_neighbors=int(n_neighbors), min_dist=float(min_dist), hub_num=int(hub_num)).fit_transform(data)
						return extract_score(zadu_obj, embedding)
					except:
						return -1 if is_higher_better else 100
				pbounds = {'n_neighbors': (2, 200), 'min_dist': (0.001, 0.999), 'hub_num': (50, 500)}
			elif method == "lle":
				def run_method(n_neighbors):
					try:
						n_neighbors = int(n_neighbors)
						embedding = LocallyLinearEmbedding(n_neighbors=n_neighbors).fit_transform(data)
						return extract_score(zadu_obj, embedding)
					except:
						return -1 if is_higher_better else 100
				pbounds = {'n_neighbors': (2, 200)}
			elif method == "isomap":
				def run_method(n_neighbors):
					try:
						n_neighbors = int(n_neighbors)
						embedding = Isomap(n_neighbors=n_neighbors).fit_transform(data)
						return extract_score(zadu_obj, embedding)
					except:
						return -1 if is_higher_better else 100
				pbounds = {'n_neighbors': (2, 200)}
			
			optimizer = BayesianOptimization(
				f=run_method,
				pbounds=pbounds,
				random_state=1,
				allow_duplicate_points=True
			)


			with open(f"./app_2/training/results/{trial_idx}_{method}_{DR_METRIC}.json") as f:
				data_file = json.load(f)
			predicted_acc = data_file["predictions"][dataset_idx]

			print(predicted_acc)
			if method == "pca":
				start = time.time()
				dataset_adpative_optimization_score = run_method()
				end = time.time()
				recorded_time = end - start
			else:
				start = time.time()
				for i in range(50):
					optimizer.maximize(
						init_points=1 if i < 10 else 0,
						n_iter=1 if i >= 10 else 0,
					)
					if optimizer.max["target"] >= predicted_acc:
						break
				end = time.time()

				recorded_time = end - start
				dataset_adpative_optimization_score = optimizer.max["target"]

			if method == top_1_techniques:
				top_1_times = recorded_time
				top_1_scores = dataset_adpative_optimization_score
			
			top_3_times += recorded_time
			top_3_scores = max(top_3_scores, dataset_adpative_optimization_score)

		results = {
			"dataset": dataset,
			"top_1_time": top_1_times,
			"top_1_score": top_1_scores,
			"top_3_time": top_3_times,
			"top_3_score": top_3_scores
		}

		with open(f"./app_2/pipeline/{trial_idx}_{dataset_idx}_dataset_adaptive.json", "w") as f:
			json.dump(results, f)
