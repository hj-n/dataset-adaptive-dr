# Experiment 3:  DOES OUR DATASET-ADAPTIVE WORKFLOW IMPROVE THE CONVENTIONAL WORKFLOW?

import exp.load_config as lc
import src.modules.load as l

from src.DatasetAdaptiveDR import DatasetAdaptiveDR
import time 
import numpy as np
import json, os

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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
MAX_POINTS = lc.load_config("MAX_POINTS")

COMPLEXITY_METRIC = "pdsmnc"

for dr_metrics in DR_METRICS:
	dr_metrics_id = dr_metrics["id"]
	dr_metrics_names = dr_metrics["names"]
	dr_metrics_params = dr_metrics["params"]
	is_higher_better = dr_metrics["is_higher_better"]

	top1_scores_list = []
	top1_times_list = []
	top3_scores_list = []
	top3_times_list = []
	gt_scores_list = []
	gt_times_list = []

	for i in range(REGRESSION_ITER):
		shuffled_indices = np.random.permutation(len(DATASET_LIST))
		shuffled_dataset_list = [DATASET_LIST[i] for i in shuffled_indices]

		training_dataset_names = shuffled_dataset_list[:TRAINING_SIZE]
		testing_dataset_names = shuffled_dataset_list[TRAINING_SIZE:TRAINING_SIZE + TEST_SIZE]


		dadr = DatasetAdaptiveDR(
			dr_techniques=DR_TECHNIQUES,
			dr_metric=dr_metrics_id,
			dr_metric_names=dr_metrics_names,
			params=dr_metrics_params,
			is_higher_better=is_higher_better,
			init_points=INIT_POINTS,
			n_iter=MAX_ITER,
			complexity_metric=COMPLEXITY_METRIC,
			training_info=TRAINING_INFO
		)
		for dataset_name in tqdm(training_dataset_names):
			data, labels = l.load_dataset(DATASET_PATH, dataset_name, data_point_maxnum=MAX_POINTS)
			dadr.add_dataset(data, labels)
		dadr.fit()

		for dataset_name in tqdm(testing_dataset_names):
			data, labels = l.load_dataset(DATASET_PATH, dataset_name, data_point_maxnum=MAX_POINTS)

			start = time.time()
			top1_score, _ = dadr.predict_opt(data, 1, labels)
			end = time.time()
			top1_time = end - start

			start = time.time()
			top3_score, _ = dadr.predict_opt(data, 3, labels) 
			end = time.time()
			top3_time = end - start

			## get ground truth score

			gt_score = -100000 if is_higher_better else 1000000
			gt_time = 0
			for dr_technique in DR_TECHNIQUES:
				with open(f"./exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metrics_id}/{dataset_name}.json") as f:
					ground_truth = json.load(f)
					gt_score = max(gt_score, ground_truth["score"]) if is_higher_better else min(gt_score, ground_truth["score"])
					gt_time += ground_truth["time"]

			top1_scores_list.append(top1_score)
			top1_times_list.append(top1_time)
			top3_scores_list.append(top3_score)
			top3_times_list.append(top3_time)
			gt_scores_list.append(gt_score)
			gt_times_list.append(gt_time)

		## save the results
		if not os.path.exists(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}"):
			os.makedirs(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}")
		
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top1_scores.json", "w") as f:
			json.dump(top1_scores_list, f)
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top1_times.json", "w") as f:
			json.dump(top1_times_list, f)
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top3_scores.json", "w") as f:
			json.dump(top3_scores_list, f)
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top3_times.json", "w") as f:
			json.dump(top3_times_list, f)
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/gt_scores.json", "w") as f:
			json.dump(gt_scores_list, f)
		with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/gt_times.json", "w") as f:
			json.dump(gt_times_list, f)

