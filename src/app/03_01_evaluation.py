import numpy as np
import os, json
from scipy.stats import spearmanr
import pandas as pd

DATASET_PATH = "../labeled-datasets/npy/"
DATASET_LIST = os.listdir(DATASET_PATH)


DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]


with open("./exp/scores/reducibility/mnc_50.json") as f:
	MNC_50 = json.load(f)

MNC_50_DATASETS = list(MNC_50.keys())
MNC_50_SCORES = [MNC_50[dataset]["score"] for dataset in MNC_50_DATASETS]

sorter = np.argsort(MNC_50_SCORES)
MNC_50_DATASETS = np.array(MNC_50_DATASETS)[sorter]
MNC_50_SCORES = np.array(MNC_50_SCORES)[sorter]

with open("./exp/scores/reducibility/pds.json") as f:
	PDS = json.load(f)

PDS_DATASETS = list(PDS.keys())
PDS_SCORES = [PDS[dataset]["score"] for dataset in PDS_DATASETS]

sorter = np.argsort(PDS_SCORES)
PDS_DATASETS = np.array(PDS_DATASETS)[sorter]
PDS_SCORES = np.array(PDS_SCORES)[sorter]

## make "Average ranking" of datasets

avg_ranking = []
for dataset in DATASET_LIST:
	mnc_50_rank = np.where(MNC_50_DATASETS == dataset)[0]
	pds_rank = len(PDS_DATASETS) - np.where(PDS_DATASETS == dataset)[0]
	avg_rank = np.mean([mnc_50_rank, pds_rank])
	avg_ranking.append(avg_rank)

avg_ranking = np.array(avg_ranking)
sorter = np.argsort(avg_ranking)
AVG_DATASETS = np.array(DATASET_LIST)[sorter]


def load_dataset(dataset_path):
	data = np.load(dataset_path)
	return data


def extract_ranking(datasets, dr_metric):
	avg_ranking_techniques = np.zeros(len(DR_TECHNIQUES))
	for dataset in datasets:
		scores = []
		for technique in DR_TECHNIQUES:
			with open(f"./ground_truth/results/{technique}_{dr_metric}.json") as f:
				ground_truth = json.load(f)
				scores.append(ground_truth[dataset]["score"])
		
		ranking = np.argsort(scores)
		avg_ranking_techniques += ranking
	
	avg_ranking_techniques = avg_ranking_techniques / len(datasets)
	final_ranking = np.argsort(avg_ranking_techniques)
	return final_ranking



## informations
Pzero_correlations_list = []
Pzero_metrics_list = []

Pplusminus_correlations_list = []
Pplusminus_metrics_list = []
Pplusminus_pm_list = []
Pplusminus_rankingby_list = []


for dr_metric in DR_METRICS:
	for ranking_method in ["mnc", "pds", "mncpds"]:
		if ranking_method == "mnc":
			Pplus_datasets = MNC_50_DATASETS[:20]
			Pminus_datasets = MNC_50_DATASETS[-20:]
		elif ranking_method == "pds":
			Pplus_datasets = PDS_DATASETS[:20]
			Pminus_datasets = PDS_DATASETS[-20:]
		elif ranking_method == "mncpds":
			Pplus_datasets = AVG_DATASETS[:20]
			Pminus_datasets = AVG_DATASETS[-20:]
		

		spearmanr_plue_list = []
		spearmanr_minus_list = []
		spearmanr_zero_list = []
		for idx in range(10):
			## pick random 20 datasets from the entire dataset list
			Pzero_datasets = np.random.choice(DATASET_LIST, 20, replace=False)

			Pplus_rankings = []
			Pminus_rankings = []
			Pzero_rankings = []
			for jdx in range(30):

			## pick random 10 datasets from Pplus and Pminus
				Pplus_sampled = np.random.choice(Pplus_datasets, 10, replace=False)
				Pminus_sampled = np.random.choice(Pminus_datasets, 10, replace=False)
				Pzero_sampled = np.random.choice(Pzero_datasets, 10, replace=False)

				Pplus_rankings.append(extract_ranking(Pplus_sampled, dr_metric))
				Pminus_rankings.append(extract_ranking(Pminus_sampled, dr_metric))
				Pzero_rankings.append(extract_ranking(Pzero_sampled, dr_metric))
			
			for ii in range(30):
				for jj in range(ii + 1, 30):
					spearmanr_plus, _ = spearmanr(Pplus_rankings[ii], Pplus_rankings[jj])
					spearmanr_minus, _ = spearmanr(Pminus_rankings[ii], Pminus_rankings[jj])
					spearmanr_zero, _ = spearmanr(Pzero_rankings[ii], Pzero_rankings[jj])

					spearmanr_plue_list.append(spearmanr_plus)
					spearmanr_minus_list.append(spearmanr_minus)
					spearmanr_zero_list.append(spearmanr_zero)
		
		Pzero_correlations_list += spearmanr_zero_list
		Pzero_metrics_list += [dr_metric] * len(spearmanr_zero_list)

		Pplusminus_correlations_list += spearmanr_plue_list + spearmanr_minus_list
		Pplusminus_metrics_list += [dr_metric] * (len(spearmanr_plue_list) + len(spearmanr_minus_list))
		Pplusminus_pm_list += ["plus"] * len(spearmanr_plue_list) + ["minus"] * len(spearmanr_minus_list)
		Pplusminus_rankingby_list += [ranking_method] * (len(spearmanr_plue_list) + len(spearmanr_minus_list))


df_pplusminus = pd.DataFrame({
	"metric": Pplusminus_metrics_list,
	"correlation": Pplusminus_correlations_list,
	"pm": Pplusminus_pm_list,
	"rankingby": Pplusminus_rankingby_list
})

df_pzero = pd.DataFrame({
	"metric": Pzero_metrics_list,
	"correlation": Pzero_correlations_list
})

df_pplusminus.to_csv("./app/results/exp2_pplusminus.csv", index=False)

df_pzero.to_csv("./app/results/exp2_pzero.csv", index=False)
