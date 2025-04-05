# Experiment 1: IS PDS+MNC VALID STRUCTURAL COMPLEXITY METRICS?

# Approximating ground truth structural complexity

from tqdm import tqdm
import time
import src.modules.opt_conv as ocv
import exp.load_config as lc
import src.modules.load as l


## Hyperparameters

DATASET_PATH = lc.load_config("DATASET_PATH")
MAX_POINTS = lc.load_config("MAX_POINTS")
DR_TECHNIQUES = lc.load_config("DR")


## 1단계: 데이터셋을 불러온다. 

dataset_names = l.load_names(DATASET_PATH)

for 

for dataset_name in dataset_names:
	data, labels = l.load_dataset(
		dataset_path=DATASET_PATH,
		dataset_name=dataset_name,
		data_point_maxnum=MAX_POINTS
	)




## 2단계: 각 데이터셋에 대해, structural complexity를 계산한다. 


def ground_truth_structural_complexity(
		data,
		labels,
		method,
		measure,
		measure_names, 
		params, 
		is_higher_better=True
):
	# scores_dict = {}
	# for dataset in tqdm(DATASET_LIST):
	# 	## TODO: 이부분을 다른 파일로 빼야함
	# 	data = hp.load_dataset(os.path.join(DATASET_PATH, dataset +  "/data.npy"))
	# 	labels = hp.load_dataset(os.path.join(DATASET_PATH, dataset +  "/label.npy"))
	# 	if data.shape[0] > DATA_POINT_MAXNUM:
	# 		filterer= np.random.choice(data.shape[0], DATA_POINT_MAXNUM, replace=False)
	# 		data = data[filterer]
	# 		labels = labels[filterer]
	start = time.time()
	score, embeeding_params = ocv.opt_conv(data, labels, method, measure, measure_names, params, is_higher_better)
	end = time.time()
	time_taken = end - start
	return {"score": score, "params": embeeding_params, "time": time_taken}

	# print("Saving scores to file...")
	# if not os.path.exists("./ground_truth/results"):
	# 	os.makedirs("./ground_truth/results")
	# with open(f"./ground_truth/results/{method}_{measure}_{identifier}.json", "w") as f:
	# 	json.dump(scores_dict, f)
