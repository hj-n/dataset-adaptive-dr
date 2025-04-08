from .metrics.mnc import mutual_neighbor_consistency as mnc
from .metrics.pds import pairwise_distance_shift as pds
from .intrinsic_dim.geometric import intrinsic_dim_geometric as intdim_geo
from .intrinsic_dim.projection import intrinsic_dim_projection as intdim_proj

from .modules.opt_conv import opt_conv
from .modules.train import train

import numpy as np

class DatasetAdaptiveDR:
	def __init__(
		self, 
		dr_techniques,
		dr_metric,
		dr_metric_names,
		params,
		is_higher_better,
		init_points,
		n_iter,
		complexity_metric,
		training_info = {
			"single_task_time": 30,
			"total_task_time": 600,
			"memory_limit": 10000,
			"cv_fold": 5
		}
	):
		self.dr_techniques = dr_techniques
		if len(set(dr_techniques).intersection({"pca", "umap", "tsne", "umato", "lle", "isomap"})) != len(dr_techniques):
			raise ValueError("dr_techniques must be a subset of ['pca', 'umap', 'tsne', 'umato', 'lle', 'isomap']")

		self.dr_metric = dr_metric
		if dr_metric not in ["tnc", "mrre", "l_tnc", "srho", "pr"]:
			raise ValueError("dr_metric must be one of ['tnc', 'mrre', 'l_tnc', 'srho', 'pr']")

		self.complexity_metric = complexity_metric
		if complexity_metric not in ["mnc", "pds", "pdsmnc", "intdim_proj", "intdim_geo"]:
			raise ValueError("complexity_metric must be one of ['mnc', 'pds', 'pdsmnc', 'intdim_proj', 'intdim_geo']")
		
		self.maximum_achievable_accuracy = {}
		self.models = {}
		for technique in self.dr_techniques:
			self.maximum_achievable_accuracy[technique] = []
			self.models[technique] = None


		

		
		if self.complexity_metric != "pdsmnc":
			self.complexity_metric_scores = []
		else:
			self.complexity_metric_scores = {
				"pds": [],
				"mnc_25": [],
				"mnc_50": [],
				"mnc_75": []
			}		
		
		self.dr_metric_names = dr_metric_names
		self.params = params
		self.is_higher_better = is_higher_better
		self.init_points = init_points
		self.n_iter = n_iter
		self.training_info = training_info


	def add_dataset(self, data, labels=None):
		self._add_complexity_metric_score(data)
		self._add_maximum_achievable_accuracy(data, labels=labels)
		pass

	def fit(self):
		if self.complexity_metric == "pdsmnc":
			self.source = np.array([
				self.complexity_metric_scores["pds"],
				self.complexity_metric_scores["mnc_25"],
				self.complexity_metric_scores["mnc_50"],
				self.complexity_metric_scores["mnc_75"]
			]).T
		else:
			self.source = np.array(self.complexity_metric_scores).reshape(-1, 1)
		

		for dr_technique in self.dr_techniques:
			self.models[dr_technique] = train(
				self.source,
				self.maximum_achievable_accuracy[dr_technique],
				self.training_info
			)
	
	def predict(self, data):
		if self.complexity_metric == "pdsmnc":
			source = np.array([
				mnc(data, 25),
				mnc(data, 50),
				mnc(data, 75),
				pds(data)
			]).reshape(1, -1)
		elif self.complexity_metric == "intdim_proj":
			source = np.array([intdim_proj(data)]).reshape(1, -1)
		elif self.complexity_metric == "intdim_geo":
			source = np.array([intdim_geo(data)]).reshape(1, -1)
		elif self.complexity_metric == "mnc":
			source = np.array([mnc(data, 25)]).reshape(1, -1)
		elif self.complexity_metric == "pds":
			source = np.array([pds(data)]).reshape(1, -1)

		
		results = {}
		for dr_technique in self.dr_techniques:
			if self.models[dr_technique] is None:
				raise ValueError(f"Model for {dr_technique} is not trained yet.")
			pred = self.models[dr_technique].predict(source)
			results[dr_technique] = pred
		return results
			


	def _add_complexity_metric_score(self, data):
		if self.complexity_metric == "mnc":
			score = mnc(data, 25)
			self.complexity_metric_scores.append(score)
		elif self.complexity_metric == "pds":
			score = pds(data)
			self.complexity_metric_scores.append(score)
		elif self.complexity_metric == "intdim_proj":
			score = intdim_proj(data)
			self.complexity_metric_scores.append(score)
		elif self.complexity_metric == "intdim_geo":
			score = intdim_geo(data)
			self.complexity_metric_scores.append(score)
		elif self.complexity_metric == "pdsmnc":
			pds_score = pds(data)
			mnc_25_score = mnc(data, 25)
			mnc_50_score = mnc(data, 50)
			mnc_75_score = mnc(data, 75)
			self.complexity_metric_scores["pds"].append(pds_score)
			self.complexity_metric_scores["mnc_25"].append(mnc_25_score)
			self.complexity_metric_scores["mnc_50"].append(mnc_50_score)
			self.complexity_metric_scores["mnc_75"].append(mnc_75_score)
	
	def _add_maximum_achievable_accuracy(self, data, labels=None):
		for dr_technique in self.dr_techniques:
			opt_score, _ = opt_conv(
				data,
				dr_technique,
				self.dr_metric,
				self.dr_metric_names,
				self.params,
				self.init_points,
				self.n_iter,
				self.is_higher_better,
				labels=labels
			)
			self.maximum_achievable_accuracy[dr_technique].append(opt_score)
