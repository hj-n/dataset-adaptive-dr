import numpy as np

import json 

import exp.load_config as lc
import src.modules.load as l

import warnings
warnings.filterwarnings("ignore")

DR_METRICS = lc.load_config("METRICS")

for dr_metrics in DR_METRICS:
	dr_metrics_id = dr_metrics["id"]

	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top1_scores.json", "r") as f:
		top1_scores_file = json.load(f)
		top1_scores = []
		for top1_score_singlefile in top1_scores_file:
			key_name = list(top1_score_singlefile.keys())[0]
			top1_score = top1_score_singlefile[key_name]
			top1_scores.append(top1_score)
		
	
	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top1_times.json", "r") as f:
		top1_times = json.load(f)
	
	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top3_scores.json", "r") as f:
		top3_scores_file = json.load(f)
		top3_scores = []
		for top3_score_singlefile in top3_scores_file:
			top_score = -1000000
			for key_name in top3_score_singlefile:
				top_score = max(top_score, top3_score_singlefile[key_name])
			top3_scores.append(top_score)
		

	
	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/top3_times.json", "r") as f:
		top3_times = json.load(f)
	
	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/gt_scores.json", "r") as f:
		gt_scores = json.load(f)
	
	with open(f"./exp/exp3_workflow_comparison/results/{dr_metrics_id}/gt_times.json", "r") as f:
		gt_times = json.load(f)
	

	print("DR METRICS:", dr_metrics_id)
	print("Top1 Scores:", np.mean(top1_scores), "pm", np.std(top1_scores))
	print("Top3 Scores:", np.mean(top3_scores), "pm", np.std(top3_scores))
	print("GT Scores:", np.mean(gt_scores), "pm", np.std(gt_scores))
	print("Top1 Times:", np.mean(top1_times), "pm", np.std(top1_times))
	print("Top3 Times:", np.mean(top3_times), "pm", np.std(top3_times))
	print("GT Times:", np.mean(gt_times), "pm", np.std(gt_times))