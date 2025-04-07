import pandas as pd 
import exp.load_config as lc
import src.modules.load as l
import json
import numpy as np



DR_TECHNIQUES = lc.load_config("DR")
DR_METRICS = lc.load_config("METRICS")
DR_METRIC_ID = lc.load_config("METRICS")[0]["id"]

DATASET_PATH = lc.load_config("DATASET_PATH")
DATASET_LIST = l.load_names(DATASET_PATH)

REGRESSION_MODELS = {
	"linear": "Linear Regression",
	"polynomial": "Polynomial Regression",
	"knn": "KNN",
	"rf": "Random Forest",
	"gb": "Gradient Boosting"
}


METRICS = ["mnc", "pds", "pdsmnc", "geometric_intdim", "projection_intdim", "dr_ensemble"]

mnc_time          = []
pds_time          = []
pdsmnc_time       = []
geometric_intdim_time  = []
projection_intdim_time = []
dr_ensemble_time   = []

#### TIME ####

## mnc
with open(f"exp/exp1_metrics/results/metrics/mnc_25.json") as f:
	mnc_time_json = json.load(f)
	for dataset in DATASET_LIST:
		mnc_time.append(mnc_time_json[dataset]["time"])

## pds
with open(f"exp/exp1_metrics/results/metrics/pds.json") as f:
	pds_time_json = json.load(f)
	for dataset in DATASET_LIST:
		pds_time.append(pds_time_json[dataset]["time"])

## pdsmnc
with open("exp/exp1_metrics/results/metrics/mnc_25.json") as f:
	mnc_25_time_json = json.load(f)
with open("exp/exp1_metrics/results/metrics/mnc_50.json") as f:
	mnc_50_time_json = json.load(f)
with open("exp/exp1_metrics/results/metrics/mnc_75.json") as f:
	mnc_75_time_json = json.load(f)
with open("exp/exp1_metrics/results/metrics/pds.json") as f:
	pds_time_json = json.load(f)

for dataset in DATASET_LIST:
	pdsmnc_time.append(
		mnc_25_time_json[dataset]["time"] +
		mnc_50_time_json[dataset]["time"] +
		mnc_75_time_json[dataset]["time"] +
		pds_time_json[dataset]["time"]
	)

### geometric_intimd
with open(f"exp/exp1_metrics/results/metrics/geometric_intdim.json") as f:
	geometric_intimd_json = json.load(f)
	for dataset in DATASET_LIST:
		geometric_intdim_time.append(geometric_intimd_json[dataset]["time"])

### projection_intimd
with open(f"exp/exp1_metrics/results/metrics/projection_intdim.json") as f:
	projection_intimd_json = json.load(f)
	for dataset in DATASET_LIST:
		projection_intdim_time.append(projection_intimd_json[dataset]["time"])


### DR Ensemble

for dataset in DATASET_LIST:
	curr_time = 0
	for dr_techniques in DR_TECHNIQUES:
		with open(f"exp/exp1_metrics/results/ground_truth/{dr_techniques}/{DR_METRIC_ID}/{dataset}.json") as f:
			dr_ensemble_time_json = json.load(f)
			curr_time += dr_ensemble_time_json["time"]
	dr_ensemble_time.append(curr_time)

print("==== Average RUNTIME for each metric ====")
print("mnc: ", np.mean(mnc_time), "+-", np.std(mnc_time))
print("pds: ", np.mean(pds_time), "+-", np.std(pds_time))
print("pdsmnc: ", np.mean(pdsmnc_time), "+-", np.std(pdsmnc_time))
print("geometric_intimd: ", np.mean(geometric_intdim_time), "+-", np.std(geometric_intdim_time))
print("projection_intimd: ", np.mean(projection_intdim_time), "+-", np.std(projection_intdim_time))
print("dr_ensemble: ", np.mean(dr_ensemble_time), "+-", np.std(dr_ensemble_time))

## save results

df = pd.DataFrame({
	"dataset": DATASET_LIST,
	"mnc_time": mnc_time,
	"pds_time": pds_time,
	"pdsmnc_time": pdsmnc_time,
	"geometric_intimd_time": geometric_intdim_time,
	"projection_intimd_time": projection_intdim_time,
	"dr_ensemble_time": dr_ensemble_time
})
df.to_csv("exp/exp1_metrics/results/final/runtime.csv", index=False)

#### SCORE ###

scores_df = pd.read_csv("exp/exp1_metrics/results/final/correlations.csv")

METRICS = ["mnc_25", "mnc_50", "mnc_75", "pds", "pdsmnc", "geometric_intdim", "projection_intdim"]


print()
print("==== Predictive Power for each metric towards Ground truth====")
for dr_metric in DR_METRICS:
	print("Ground truth based on: ", dr_metric["id"])
	for metric in METRICS:
		print("- metric: ", metric)
		for regression_model in REGRESSION_MODELS:
			row = scores_df[
				(scores_df["dr_metric"] == dr_metric["id"]) &
				(scores_df["competitor"] == metric) &
				(scores_df["regression_model"] == regression_model)
			]
			print("--- ", REGRESSION_MODELS[regression_model], ": ", row["r2"].values[0])
