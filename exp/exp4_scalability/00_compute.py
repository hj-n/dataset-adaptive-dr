import numpy as np
import os, json
import time
from tqdm import tqdm

from src.metrics.mnc import mutual_neighbor_consistency as mnc
from src.metrics.pds import pairwise_distance_shift as pds

from numba import cuda
cuda.select_device(1)

import exp.load_config as lc
import src.modules.load as l

from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_rcv1

import pandas as pd

import umap
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap, LocallyLinearEmbedding

import umato


kddcup99_data = fetch_kddcup99(return_X_y=True)[0]
covtype_data = fetch_covtype(return_X_y=True)[0]
rcv1_data = fetch_rcv1(return_X_y=True)[0]

## filter out categorical features
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
           'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
           'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
           'num_access_files', 'num_outbound_cmds', 'is_host_login',
           'is_guest_login', 'count', 'srv_count', 'serror_rate',
           'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
           'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate']

# Identify categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']

kddcup99_data = pd.DataFrame(kddcup99_data, columns=columns)
kddcup99_numerical_data = kddcup99_data.drop(columns=categorical_cols)

kddcup99_data = kddcup99_numerical_data.to_numpy()


datasets = [
	covtype_data.astype(np.float32),
	kddcup99_data.astype(np.float32),
	rcv1_data.astype(np.float32)
]


dataset_names = [
	"covtype",
	"kddcup99",
	"rcv1"
]


def return_time(func, data):
	start = time.time()
	score = func(data)
	end = time.time()
	return end - start

times_mnc = {}
times_pds = {}
times_umap = {}
times_tsne = {}
times_isomap = {}
times_lle = {}
times_umato = {}
percentages = {}

for i, dataset in enumerate(datasets):
	dataset_name = dataset_names[i]
	print(f"Processing {dataset_name}...")

	times_mnc[dataset_name] = []
	times_pds[dataset_name] = []
	times_umap[dataset_name] = []
	times_tsne[dataset_name] = []
	# times_isomap[dataset_name] = []
	# times_lle[dataset_name] = []
	times_umato[dataset_name] = []
	percentages[dataset_name] = []

	## sample from 2.5, 5, 7.5, .... percent of the dataset
	for percent in tqdm(np.arange(0.5, 10.0, 0.5)):
		## random pick
		print(percent)
		sampled_data = dataset[np.random.choice(dataset.shape[0], int(dataset.shape[0] * percent / 100), replace=False)]
		print(np.shape(sampled_data))

		umap_obj = umap.UMAP()
		tsne_obj = TSNE()
		# isomap_obj = Isomap()
		# lle_obj = LocallyLinearEmbedding()
		umato_obj = umato.UMATO()

		for j in range(5):
			computed_time_mnc = return_time(lambda x: mnc(x, 25), sampled_data)
			computed_time_pds = return_time(lambda x: pds(x), sampled_data)
			computed_time_umap = return_time(lambda x: umap_obj.fit_transform(x), sampled_data)
			computed_time_tsne = return_time(lambda x: tsne_obj.fit_transform(x), sampled_data)
			# computed_time_isomap = return_time(lambda x: isomap_obj.fit_transform(x), sampled_data)
			# computed_time_lle = return_time(lambda x: lle_obj.fit_transform(x), sampled_data)
			computed_time_umato = return_time(lambda x: umato_obj.fit_transform(x), sampled_data)


			times_mnc[dataset_name] += [computed_time_mnc]
			times_pds[dataset_name] += [computed_time_pds]
			times_umap[dataset_name] += [computed_time_umap]
			times_tsne[dataset_name] += [computed_time_tsne]
			# times_isomap[dataset_name] += [computed_time_isomap]
			# times_lle[dataset_name] += [computed_time_lle]
			times_umato[dataset_name] += [computed_time_umato]

			percentages[dataset_name] += [percent]


	print("Saving results...")
	if not os.path.exists("exp/exp4_scalability/results/"):
		os.makedirs("exp/exp4_scalability/results/")
	with open(f"exp/exp4_scalability/results/times_{dataset_name}.json", "w") as f:
		json.dump({
			"mnc": times_mnc[dataset_name],
			"pds": times_pds[dataset_name],
			"umap": times_umap[dataset_name],
			"tsne": times_tsne[dataset_name],
			"umato": times_umato[dataset_name],
			"percentages": percentages[dataset_name]
		}, f, indent=4)