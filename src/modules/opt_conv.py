from zadu import zadu
from bayes_opt import BayesianOptimization
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

import umato


def opt_conv(
	data, 									## high-dimensional data
	method, 						  	## supported DR methods: "pca", "umap", "tsne", "trimap", "isomap", "lle", "umato"
	measure, 								## the id of the measure to be used for evaluation; specified in the zadu library
	measure_names, 					## the names of the measures to be used for evaluation; specified in the zadu library
	params, 								## parameter for the measure; specified in the zadu library
	init_points=10, 				## number of initial points to sample; used for Bayesian optimization
	n_iter=40, 							## number of iterations to sample; used for Bayesian optimization
	is_higher_better=True,  ## True if the higher the better; False if the lower the better
	labels = None 					## labels for the data points; used for evaluation if needed
):
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
		allow_duplicate_points=True,
		verbose=0
	)

	optimizer.maximize(init_points=init_points, n_iter=n_iter)

	score = optimizer.max['target'] if is_higher_better else -optimizer.max['target']
	opt_params = optimizer.max['params']

	return score, opt_params

