import numpy as np
import autosklearn.regression
import os, json
from bayes_opt import BayesianOptimization


def train(
	source,
	target,
	training_info,
):
	automl = autosklearn.regression.AutoSklearnRegressor(
		time_left_for_this_task=training_info["total_task_time"],                   
		per_run_time_limit=training_info["single_task_time"],
		resampling_strategy='cv',
		memory_limit=training_info["memory_limit"],
		resampling_strategy_arguments={'folds': 5}
	)

	automl.fit(source, target)

	return automl