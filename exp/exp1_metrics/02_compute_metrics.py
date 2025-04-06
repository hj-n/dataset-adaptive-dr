import numpy as np
import os, json
import time
from tqdm import tqdm

from src.metrics.mnc import mutual_neighbor_consistency as mnc
from src.metrics.pds import pairwise_distance_shift as pds

from exp.intrinsic_dim.geometric import intrinsic_dim_geometric as intdim_geo
from exp.intrinsic_dim.projection import intrinsic_dim_projection as intdim_proj

from numba import cuda
cuda.select_device(1)

import exp.load_config as lc
import src.modules.load as l


DATASET_PATH = lc.load_config("DATASET_PATH")
DATASET_LIST = l.load_names(DATASET_PATH)

DATA_POINT_MAXNUM = lc.load_config("MAX_POINTS")




## STEP 1: Compute the reducibility and intrinsic dimensionality 

mnc_25_scores = []
mnc_50_scores = []
mnc_75_scores = []
pds_scores = []
projection_intdim_scores = []
geometric_intdim_scores = []

mnc_25_times = []
mnc_50_times = []
mnc_75_times = []
pds_times = []
projection_intdim_times = []
geometric_intdim_times = []


def return_score_and_time(func, data):
	start = time.time()
	score = func(data)
	end = time.time()
	return end - start, score


for dataset_name in tqdm(DATASET_LIST):

	data, labels = l.load_dataset(
		dataset_path=DATASET_PATH,
		dataset_name=dataset_name,
		data_point_maxnum=DATA_POINT_MAXNUM 
	)

	timed, scores = return_score_and_time(lambda x: mnc(x, 25), data)
	mnc_25_scores.append(scores)
	mnc_25_times.append(timed)


	timed, scores = return_score_and_time(lambda x: mnc(x, 50), data)
	mnc_50_scores.append(scores)
	mnc_50_times.append(timed)


	timed, scores = return_score_and_time(lambda x: mnc(x, 75), data)
	mnc_75_scores.append(scores)
	mnc_75_times.append(timed)


	timed, scores = return_score_and_time(lambda x: pds(x), data)
	pds_scores.append(scores)
	pds_times.append(timed)

	timed, scores = return_score_and_time(lambda x: intdim_proj(x, 1), data)
	projection_intdim_scores.append(scores)
	projection_intdim_times.append(timed)

	timed, scores = return_score_and_time(lambda x: intdim_geo(x, 1), data)
	geometric_intdim_scores.append(scores)
	geometric_intdim_times.append(timed)
	

def save_result(scores, times, path):
	result = {}
	for i in range(len(DATASET_LIST)):
		result[DATASET_LIST[i]] = {
			"score": scores[i],
			"time": times[i]
		}
	with open(path, "w") as f:
		json.dump(result, f)


if not os.path.exists("./exp/exp1_metrics/results/metrics/"):
	os.makedirs("./exp/exp1_metrics/results/metrics/")


save_result(mnc_25_scores, mnc_25_times, "./exp/exp1_metrics/results/metrics/mnc_25.json")
save_result(mnc_50_scores, mnc_50_times, "./exp/exp1_metrics/results/metrics/mnc_50.json")
save_result(mnc_75_scores, mnc_75_times, "./exp/exp1_metrics/results/metrics/mnc_75.json")
save_result(pds_scores, pds_times, "./exp/exp1_metrics/results/metrics/pds.json")
save_result(projection_intdim_scores, projection_intdim_times, "./exp/exp1_metrics/results/metrics/projection_intdim.json")
save_result(geometric_intdim_scores, geometric_intdim_times, "./exp/exp1_metrics/results/metrics/geometric_intdim.json")

