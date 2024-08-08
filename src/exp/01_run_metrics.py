import numpy as np
import os, json
import time
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reducibility.mnc import mutual_neighbor_consistency as mnc
from reducibility.pds import pairwise_distance_shift as pds

from intrinsic_dim.embedding_based import intrinsic_dim_pca as intrinsic_dim_emb
from intrinsic_dim.geometric import intrinsic_dim_correlation as intrinsic_dim_geo

from numba import cuda
cuda.select_device(1)



DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DATA_POINT_MAXNUM = 2000


def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data




## STEP 1: Compute the reducibility and intrinsic dimensionality 

mnc_25_scores = []
mnc_50_scores = []
mnc_75_scores = []
pds_scores = []
embedding_intdim_scores = []
geometric_intdim_scores = []

mnc_25_times = []
mnc_50_times = []
mnc_75_times = []
pds_times = []
embedding_intdim_times = []
geometric_intdim_times = []


def return_score_and_time(func, data):
	start = time.time()
	score = func(data)
	end = time.time()
	return end - start, score


for dataset in tqdm(DATASET_LIST):
	data = load_dataset(os.path.join(DATASET_PATH, dataset + "/data.npy"))
	labels = load_dataset(os.path.join(DATASET_PATH, dataset + "/label.npy"))
	if data.shape[0] > DATA_POINT_MAXNUM:
		filterer = np.random.choice(data.shape[0], DATA_POINT_MAXNUM, replace=False)
		data = data[filterer]
		labels = labels[filterer]


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

	timed, scores = return_score_and_time(lambda x: intrinsic_dim_emb(x, 1), data)
	embedding_intdim_scores.append(scores)
	embedding_intdim_times.append(timed)

	timed, scores = return_score_and_time(lambda x: intrinsic_dim_geo(x, 1), data)
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


## if path does not exist, create it
if not os.path.exists("./exp/scores/reducibility"):
	os.makedirs("./exp/scores/reducibility")
if not os.path.exists("./exp/scores/intrinsic_dim"):
	os.makedirs("./exp/scores/intrinsic_dim")


save_result(mnc_25_scores, mnc_25_times, "./exp/scores/reducibility/mnc_25.json")
save_result(mnc_50_scores, mnc_50_times, "./exp/scores/reducibility/mnc_50.json")
save_result(mnc_75_scores, mnc_75_times, "./exp/scores/reducibility/mnc_75.json")
save_result(pds_scores, pds_times, "./exp/scores/reducibility/pds.json")
save_result(embedding_intdim_scores, embedding_intdim_times, "./exp/scores/intrinsic_dim/embedding_intdim.json")
save_result(geometric_intdim_scores, geometric_intdim_times, "./exp/scores/intrinsic_dim/geometric_intdim.json")

