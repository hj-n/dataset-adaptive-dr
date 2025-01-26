
import json

DR_METRICS = ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]
DR_TECHNIQUES = ["umap", "tsne", "pca", "lle", "isomap", "umato"]

for metric in DR_METRICS:
	top_3_accuracy_list = []
	top_1_accuracy_list = []
	for trial in range(10,22):

		for dataset_idx in range(16):
			predictions = {}
			ground_truths = {}
			for technique in DR_TECHNIQUES:
				with open(f"./app_2/training/results/{trial}_{technique}_{metric}.json") as f:
					data = json.load(f)
					predictions[technique] = data["predictions"][dataset_idx]
					ground_truths[technique] = data["true"][dataset_idx]

				top_3_ground_truths = sorted(ground_truths.items(), key=lambda x: x[1], reverse=True)[:3]
				top_3_ground_truths = [x[0] for x in top_3_ground_truths]
				top_1_ground_truths = sorted(ground_truths.items(), key=lambda x: x[1], reverse=True)[:1]
				top_1_ground_truths = [x[0] for x in top_1_ground_truths]

				top_1_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:1]
				top_1_predictions = [x[0] for x in top_1_predictions]

				top_1_ground_truths = set(top_1_ground_truths)
				top_3_ground_truths = set(top_3_ground_truths)
				top_1_predictions = set(top_1_predictions)
				
				top_3_accuracy = 1 if top_3_ground_truths.intersection(top_1_predictions) else 0
				top_1_accuracy = 1 if top_1_ground_truths.intersection(top_1_predictions) else 0

				top_3_accuracy_list.append(top_3_accuracy)
				top_1_accuracy_list.append(top_1_accuracy)
	
	print(f"{metric} top 3 accuracy: {sum(top_3_accuracy_list) / len(top_3_accuracy_list)}")
	print(f"{metric} top 1 accuracy: {sum(top_1_accuracy_list) / len(top_1_accuracy_list)}")