import numpy as np 
import os, json
import helpers as hp
from zadu import zadu
from bayes_opt import BayesianOptimization
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
from sklearn.manifold import Isomap, LocallyLinearEmbedding

import umato

DATASET_PATH = "../../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)
INIT_POINTS = 10
N_ITER = 40
DATA_POINT_MAXNUM = 2000


def find_best_embedding_score(data, labels, method, measure, measure_names, params, is_higher_better=True):
	spec = [{ "id": measure, "params": params }]
	zadu_obj = zadu.ZADU(spec, data)

	def extract_score(z_obj, embedding):
		score_dict = z_obj.measure(embedding, labels)[0]
		score_list = [score_dict[measure_name] for measure_name in measure_names]
		score = 2 * (score_list[0] * score_list[1]) / (score_list[0] + score_list[1]) if len(score_list) == 2 else score_list[0]
		return score if is_higher_better else -score
	
	if method == "pca":
		embedding = PCA(n_components=2).fit_transform(data)
		return extract_score(zadu_obj, embedding), {}

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
	elif method == "trimap":
		def run_method(n_inliers, n_outliers):
			n_inliers = int(n_inliers)
			n_outliers = int(n_outliers)
			embedding = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers).fit_transform(data)
			return extract_score(zadu_obj, embedding)
		pbounds = {'n_inliers': (2, 200), 'n_outliers': (2, 200)}
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

	optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)

	score = optimizer.max['target'] if is_higher_better else -optimizer.max['target']
	embeeding_params = optimizer.max['params']

	return score, embeeding_params




def save_best_embedding_score_dict(method, measure, measure_names, params, identifier, is_higher_better=True):
	scores_dict = {}
	print("Running for method:", method, "measure:", measure, "params:", params)
	for dataset in tqdm(DATASET_LIST):
		data = hp.load_dataset(os.path.join(DATASET_PATH, dataset +  "/data.npy"))
		labels = hp.load_dataset(os.path.join(DATASET_PATH, dataset +  "/label.npy"))
		if data.shape[0] > DATA_POINT_MAXNUM:
			filterer= np.random.choice(data.shape[0], DATA_POINT_MAXNUM, replace=False)
			data = data[filterer]
			labels = labels[filterer]
		start = time.time()
		score, embeeding_params = find_best_embedding_score(data, labels, method, measure, measure_names, params, is_higher_better)
		end = time.time()
		time_taken = end - start
		scores_dict[dataset] = {"score": score, "params": embeeding_params, "time": time_taken}

	print("Saving scores to file...")
	if not os.path.exists("./results"):
		os.makedirs("./results")
	with open(f"./results/{method}_{measure}_{identifier}.json", "w") as f:
		json.dump(scores_dict, f)


dr_methods = ["pca", "umap", "tsne", "umato", "lle", "isomap"]

for dr_method in dr_methods:
	print("Running for method:", dr_method)

	save_best_embedding_score_dict(
		dr_method, "mrre", ["mrre_false", "mrre_missing"], {"k": 25}, 25, True
	)
	save_best_embedding_score_dict(
		dr_method, "l_tnc", ["label_trustworthiness", "label_continuity"], {}, 0, True
	)
	save_best_embedding_score_dict(
		dr_method, "tnc", ["trustworthiness", "continuity"], {"k": 25}, 25, True
	)
	save_best_embedding_score_dict(
		dr_method, "srho", ["spearman_rho"], {}, 0, True
	)

	save_best_embedding_score_dict(
		dr_method, "pr", ["pearson_r"], {}, 0, True
	)

