# Experiment 1: IS PDS+MNC VALID STRUCTURAL COMPLEXITY METRICS?

# Approximating ground truth structural complexity

from tqdm import tqdm
import time
import src.modules.opt_conv as ocv
import exp.load_config as lc
import src.modules.load as l

import os, json


## Hyperparameters

DATASET_PATH = lc.load_config("DATASET_PATH")
MAX_POINTS = lc.load_config("MAX_POINTS")
DR_TECHNIQUES = lc.load_config("DR")
DR_METRICS = lc.load_config("METRICS")
INIT_POINTS = lc.load_config("INIT_POINTS")
MAX_ITER = lc.load_config("MAX_ITER")


## functions


def ground_truth_structural_complexity(
		data,
		labels,
		method,
		measure,
		measure_names, 
		params, 
		is_higher_better=True
):
	start = time.time()
	score, embeeding_params = ocv.opt_conv(
		data, 
		method, 
		measure, 
		measure_names, 
		params, 
		init_points=INIT_POINTS,
		n_iter=MAX_ITER,
		is_higher_better=is_higher_better,
		labels=labels
	)
	end = time.time()
	time_taken = end - start
	return {"score": score, "params": embeeding_params, "time": time_taken}


## 1단계: 데이터셋을 불러온다. 

dataset_names = l.load_names(DATASET_PATH)

print("DR techniques: ", ", ".join(DR_TECHNIQUES))
print("DR metrics: ", ", ".join([dr_metric_info["id"] for dr_metric_info in DR_METRICS]))
print("Number of total combinations of DR techniques and metrics: ", len(DR_TECHNIQUES) * len(DR_METRICS))

for ii, dr_technique in enumerate(DR_TECHNIQUES):
	if not os.path.exists(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/"):
		os.makedirs(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/")

	for jj, dr_metric_info in enumerate(DR_METRICS):
		if not os.path.exists(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metric_info['id']}/"):
			os.makedirs(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metric_info['id']}/")
		print(f"Calculating {dr_metric_info['id']} for {dr_technique}...")
		print("Progress: ", f"{ii * len(DR_METRICS) + jj + 1}/{len(DR_TECHNIQUES) * len(DR_METRICS)}")
		for dataset_name in tqdm(dataset_names):
			
			if os.path.exists(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metric_info['id']}/{dataset_name}.json"):
				continue

			data, labels = l.load_dataset(
				dataset_path=DATASET_PATH,
				dataset_name=dataset_name,
				data_point_maxnum=MAX_POINTS
			)
			results = ground_truth_structural_complexity(
				data=data,
				labels=labels,
				method=dr_technique,
				measure=dr_metric_info["id"],
				measure_names=dr_metric_info["names"],
				params=dr_metric_info["params"] if "params" in dr_metric_info else {},
				is_higher_better=dr_metric_info["is_higher_better"]
			)

			# Save the results
			with open(f"exp/exp1_metrics/results/ground_truth/{dr_technique}/{dr_metric_info['id']}/{dataset_name}.json", "w") as f:
				json.dump(results, f)

