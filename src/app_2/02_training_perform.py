import numpy as np 
import os, json

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

indices = range(22)


for technique in DR_TECHNIQUES:
	for metric in DR_METRICS:
		max_score = -100
		for trial in indices:
			with open(f"./app_2/training/results/{trial}_{technique}_{metric}.json") as f:
				data = json.load(f)
				max_score = max(max_score, data["score"])
		print(f"{technique} {metric} {max_score}")
	
