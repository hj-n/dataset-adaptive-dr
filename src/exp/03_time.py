import numpy as np

import os, json
from tqdm import tqdm

import pandas as pd

DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

SOURCES = [
	"/intrinsic_dim/embedding_intdim",
	"/intrinsic_dim/geometric_intdim",
	"/reducibility/mnc_50",
	"/reducibility/pds",
]


def load_data(dataset):
	data = np.load(DATASET_PATH + dataset)
	return data


ground_truth_time = []
for dataset in tqdm(DATASET_LIST):
	time_sum = 0
	for technique in DR_TECHNIQUES:
		for metric in DR_METRICS:
			with open(f"./ground_truth/results/{technique}_{metric}.json") as f:
				ground_truth = json.load(f)
				time_sum += ground_truth[dataset]["time"]
	ground_truth_time.append(time_sum / len(DR_METRICS))



emd_intdim_time = []
geo_intdim_time = []
mnc_time = []
pds_time = []

for dataset in DATASET_LIST:
	with open(f"./exp/scores/intrinsic_dim/embedding_intdim.json") as f:
		data = json.load(f)
		emd_intdim_time.append(data[dataset]["time"])

	with open(f"./exp/scores/intrinsic_dim/geometric_intdim.json") as f:
		data = json.load(f)
		geo_intdim_time.append(data[dataset]["time"])

	with open(f"./exp/scores/reducibility/mnc_50.json") as f:
		data = json.load(f)
		mnc_time.append(data[dataset]["time"])

	with open(f"./exp/scores/reducibility/pds.json") as f:
		data = json.load(f)
		pds_time.append(data[dataset]["time"])



## convert into pandas dataframe

time = []
metric = []

time += ground_truth_time
metric += ["ground_truth"] * len(ground_truth_time)

time += emd_intdim_time
metric += ["embedding_intdim"] * len(emd_intdim_time)

time += geo_intdim_time
metric += ["geometric_intdim"] * len(geo_intdim_time)

time += mnc_time
metric += ["mnc_50"] * len(mnc_time)

time += pds_time
metric += ["pds"] * len(pds_time)


df = pd.DataFrame({
	"time": time,
	"metric": metric
})

df.to_csv("./exp/data/time.csv", index=False)

