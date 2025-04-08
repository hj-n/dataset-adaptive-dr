# Experiment2 : IS PDS+MNC SUITABLE FOR THE DATASETADAPTIVE WORKFLOW?

# 1. Evaluation on pretraining regression models

import exp.load_config as lc
import src.modules.load as l

from src.DatasetAdaptiveDR import DatasetAdaptiveDR
from src.modules.opt_conv import opt_conv
from tqdm import tqdm

import numpy as np
import json, os

from sklearn.metrics import r2_score

import time

import warnings
warnings.filterwarnings("ignore")

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
MAX_POINTS = lc.load_config("MAX_POINTS")


## train the model 

COMPLEXITY_METRICS = ["pdsmnc", "intdim_proj", "intdim_geo"]

models = {}

if not os.path.exists("./exp/exp2_metrics_workflow/results/prediction/"):
	os.makedirs("./exp/exp2_metrics_workflow/results/prediction/")

for complexity_metric in COMPLEXITY_METRICS:
	print("COMPLEXITY METRIC:", complexity_metric)
	if not os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/"):
		os.makedirs(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/")
	for dr_metric in DR_METRICS:
		dr_metric_id = dr_metric["id"]
		if not os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/"):
			os.makedirs(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/")
		if os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp1_correlations.json"):
			print("CORRELATIONS ALREADY COMPUTED FOR THIS METRIC")
			if (os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp2_top_1_accuracy.json") and
				os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp2_top_3_accuracy.json")):
				print("TOP 1 AND 3 ACCURACIES ALREADY COMPUTED FOR THIS METRIC")
				if (os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_opt_time.json") and
					os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_gt_time.json") and
					os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_opt_score.json") and
					os.path.exists(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_gt_score.json")):
					print("OPTIMIZATION TIME AND SCORE ALREADY COMPUTED FOR THIS METRIC")
					continue
				
		dr_metric_names = dr_metric["names"]
		params = dr_metric["params"]
		is_higher_better = dr_metric["is_higher_better"]

		correlations_by_iter = []
		top_3_accuracies = []
		top_1_accuracies = []

		opt_scores = []
		gt_scores = []
		opt_times = []
		gt_times = []

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

			for dataset_name in tqdm(training_dataset_names):
				data, labels = l.load_dataset(DATASET_PATH, dataset_name, data_point_maxnum=MAX_POINTS)
				dadr.add_dataset(data, labels=labels)
			dadr.fit()

			predicting_results = []
			for dataset_name in testing_dataset_names:
				data, labels = l.load_dataset(DATASET_PATH, dataset_name, data_point_maxnum=MAX_POINTS)
				result = dadr.predict(data)
				predicting_results.append(result)
			
			## Evaluation 1: compute correlations between the predicted and ground truth
			ground_truth = []
			for dataset_name in testing_dataset_names:
				ground_truth_dataset = {}
				for technique in DR_TECHNIQUES:
					with open(f"./exp/exp1_metrics/results/ground_truth/{technique}/{dr_metric_id}/{dataset_name}.json") as f:
						ground_truth_dataset[technique] = json.load(f)["score"]
				ground_truth.append(ground_truth_dataset)

			
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

			correlations_by_iter.append(r2_correlations)


			## Evaluation 2: check whether the top 1 and 3 techniques in the predicted and the ground truth are the same
			ground_truth_top_1 = []
			ground_truth_top_3 = []
			prediction_top_1 = []
			prediction_top_3 = []

			for dataset_name in testing_dataset_names:
				ground_truth_dataset = {}
				prediction_dataset = {}
				for technique in DR_TECHNIQUES:
					with open(f"./exp/exp1_metrics/results/ground_truth/{technique}/{dr_metric_id}/{dataset_name}.json") as f:
						ground_truth_dataset[technique] = json.load(f)["score"]
					prediction_dataset[technique] = predicting_results[i][technique]
				
				## top 1 and 3 DR techniques, not dataset (only extract kesy)
				ground_truth_1_sorted = sorted(ground_truth_dataset.items(), key=lambda x: x[1], reverse=True)[:1]
				ground_truth_3_sorted = sorted(ground_truth_dataset.items(), key=lambda x: x[1], reverse=True)[:3]
				prediction_1_sorted = sorted(prediction_dataset.items(), key=lambda x: x[1], reverse=True)[:1]
				prediction_3_sorted = sorted(prediction_dataset.items(), key=lambda x: x[1], reverse=True)[:3]

				ground_truth_top_1.append([ground_truth_1_sorted[0][0]])
				ground_truth_top_3.append([ground_truth_3_sorted[0][0], ground_truth_3_sorted[1][0], ground_truth_3_sorted[2][0]])
				prediction_top_1.append([prediction_1_sorted[0][0]])
				prediction_top_3.append([prediction_3_sorted[0][0], prediction_3_sorted[1][0], prediction_3_sorted[2][0]])

			
			#### Measure the accuracy of the top 1 and 3
			for i in range(len(prediction_top_1)):
				curr_3_accuracy = 1 if len(set(prediction_top_3[i]) & set(ground_truth_top_3[i])) > 0 else 0
				curr_1_accuracy = 1 if len(set(prediction_top_1[i]) & set(ground_truth_top_1[i])) > 0 else 0
				top_3_accuracies.append(curr_3_accuracy)
				top_1_accuracies.append(curr_1_accuracy)
			

			## Evaluation 3: early stopping the training 
			for idx, dataset_name in enumerate(testing_dataset_names):
				data, labels = l.load_dataset(DATASET_PATH, dataset_name, data_point_maxnum=MAX_POINTS)
				for dr_technique in predicting_results[0].keys():
					print(idx, len(predicting_results))
					predicted_score = predicting_results[idx][dr_technique]
					start = time.time()
					opt_score, _ = opt_conv(
						data, 
						method=dr_technique,
						measure=dr_metric_id,
						measure_names=dr_metric["names"],
						params=dr_metric["params"],
						init_points=INIT_POINTS,
						n_iter=MAX_ITER,
						is_higher_better=dr_metric["is_higher_better"],
						labels=labels,
						early_termination=predicted_score
					)
					end = time.time()

					## get ground truth score and time
					with open(f"./exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metric_id}/{dataset_name}.json") as f:
						json_object = json.load(f)
						gt_score = json_object["score"]
						gt_time = json_object["time"]
					opt_scores.append(opt_score)
					gt_scores.append(gt_score)
					opt_times.append(end-start)
					gt_times.append(gt_time)

			

		
		## Evaluation 1: get the max correlation for each technique
		max_correlations = {}
		for technique in DR_TECHNIQUES:
			max_correlations[technique] = max([correlation[technique] for correlation in correlations_by_iter])
			print("MAX CORRELATION FOR TECHNIQUE:", technique, ":", max_correlations[technique])

		## Evaluation 2: get the top 1 and 3 accuracies
		top_1_accuracy_avg = np.mean(top_1_accuracies)
		top_3_accuracy_avg = np.mean(top_3_accuracies)

		print("TOP 1 ACCURACY:", top_1_accuracy_avg)
		print("TOP 3 ACCURACY:", top_3_accuracy_avg)
		
		## Evaluation 3: get the average score and time
		print("OPTIMIZATION TIME:", np.mean(opt_times))
		print("GROUND TRUTH TIME:", np.mean(gt_times))
		print("OPTIMIZATION SCORE:", np.mean(opt_scores))
		print("GROUND TRUTH SCORE:", np.mean(gt_scores))

		

		## save the results

		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp1_correlations.json", "w") as f:
			json.dump(max_correlations, f)

		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp2_top_1_accuracy.json", "w") as f:
			json.dump(top_1_accuracy_avg, f)
		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp2_top_3_accuracy.json", "w") as f:
			json.dump(top_3_accuracy_avg, f)


		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_opt_time.json", "w") as f:
			json.dump(np.mean(opt_times), f)
		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_gt_time.json", "w") as f:
			json.dump(np.mean(gt_times), f)
		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_opt_score.json", "w") as f:	
			json.dump(np.mean(opt_scores), f)
		with open(f"./exp/exp2_metrics_workflow/results/prediction/{complexity_metric}/{dr_metric_id}/exp3_gt_score.json", "w") as f:
			json.dump(np.mean(gt_scores), f)