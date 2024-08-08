import pandas as pd 
import numpy as np

import os, json

from scipy.stats import ttest_ind


score_original = []
score_predicted = []
time_original = []
time_predicted = []

for path in os.listdir("./app/results/final/"):
	with open(f"./app/results/final/{path}") as f:
		data = json.load(f)

		score_original.append(data["score"])
		time_original.append(data["time"])

		score_predicted.append(data["pred_score"])
		time_predicted.append(data["pred_time"])

df = pd.DataFrame({
	"score": score_original + score_predicted,
	"time": time_original + time_predicted,
	"method": ["original"]*len(score_original) + ["predicted"]*len(score_predicted)
})

tt = ttest_ind(score_original, score_predicted)
print(tt)
tt = ttest_ind(time_original, time_predicted)
print(tt)

df.to_csv("./app/results/app1_results.csv", index=False)
	