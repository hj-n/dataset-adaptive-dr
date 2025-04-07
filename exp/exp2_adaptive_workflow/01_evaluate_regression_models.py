# Experiment2 : IS PDS+MNC SUITABLE FOR THE DATASETADAPTIVE WORKFLOW?

# 1. Evaluation on pretraining regression models

import exp.load_config as lc
import src.modules.load as l

from src.DatasetAdaptiveDR import DatasetAdaptiveDR
from tqdm import tqdm

import numpy as np
import json

from sklearn.metrics import r2_score

## hyperparameters

DATASET_PATH = lc.load_config("DATASET_PATH")
DATASET_LIST = l.load_names(DATASET_PATH)
DR_TECHNIQUES = lc.load_config("DR")
DR_METRICS = lc.load_config("METRICS")
INIT_POINTS = lc.load_config("INIT_POINTS")
MAX_ITER = lc.load_config("MAX_ITER")
REGRESSION_ITER = lc.load_config("REGRESSION_ITER")
TRAINING_INFO = lc.load_config("TRAINING_INFO")
TEST_SIZE = lc.load_config("TEST_SIZE")
TRAINING_SIZE = lc.load_config("TRAINING_SIZE")


## train the model 

COMPLEXITY_METRICS = ["pdsmnc", "intdim_proj", "intdim_geo"]

models = {}

for complexity_metric in COMPLEXITY_METRICS:
	print("COMPLEXITY METRIC:", complexity_metric)
	for dr_metric in DR_METRICS:
		print("- DR METRIC:", dr_metric)
		dr_metric_id = dr_metric["id"]
		dr_metric_names = dr_metric["names"]
		params = dr_metric["params"]
		is_higher_better = dr_metric["is_higher_better"]

		for i in range(REGRESSION_ITER):
			shuffled_indices = np.random.permutation(len(DATASET_LIST))
			shuffled_dataset_list = [DATASET_LIST[i] for i in shuffled_indices]

			## split the dataset into training and testing
			training_dataset_names = shuffled_dataset_list[:TRAINING_SIZE]
			testing_dataset_names = shuffled_dataset_list[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE]

			dadr = DatasetAdaptiveDR(
				dr_techniques=DR_TECHNIQUES,
				dr_metric=dr_metric_id,
				dr_metric_names=dr_metric_names,
				params=params,
				is_higher_better=is_higher_better,
				init_points=INIT_POINTS,
				n_iter=MAX_ITER,
				complexity_metric=complexity_metric,
				training_info=TRAINING_INFO
			)

			for dataset_name in training_dataset_names:
				data, labels = l.load_dataset(dataset_name, DATASET_PATH)
				dadr.add_dataset(data, labels=labels)
			dadr.fit()

			predicting_results = []
			for dataset_name in testing_dataset_names:
				data, labels = l.load_dataset(dataset_name, DATASET_PATH)
				result = dadr.predict(data, labels=labels)
				predicting_results.append(result)

			ground_truth = []
			for dataset_name in testing_dataset_names:
				ground_truth_dataset = {}
				for technique in DR_TECHNIQUES:
					with open(f"./exp/exp1_metrics/results/ground_truth/{technique}/{dr_metric_id}/{dataset_name}.json") as f:
						ground_truth_dataset[technique] = json.load(f)["score"]
				ground_truth.append(ground_truth_dataset)

			## compute correlations between the predicted and ground truth
			r2_correlations = {}
			for technique in DR_TECHNIQUES:
				prediction = []
				truth = []
				for i in range(len(predicting_results)):
					prediction.append(predicting_results[i][technique])
					truth.append(ground_truth[i][technique])
				prediction = np.array(prediction)
				truth = np.array(truth)

				r2 = r2_score(truth, prediction)
				r2_correlations[technique] = r2
				print(" --- R2 CORRELATION FOR TECHNIQUE:", technique, ":", r2)




		