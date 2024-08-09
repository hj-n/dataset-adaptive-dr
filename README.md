<p align="center">
  <h2 align="center">Structural Reducibility Metrics</h2>
	<p align="center">Repository for the paper <i>Measuring the Structural Reducibility of Datasets for Fast and Accurate Dimensionality Reduction</i></p>
</p>

---

We introduce two structural reducibility metrics, Pairwise Distance Shift (PDS) and Mutual Neighbor Consistency (MNC), which are designed to quantify the maximum achievable structural consistency between high-dimensional (HD) datasets and their 2D projections. These metrics guide dimensionality reduction (DR) by predicting optimal iteration numbers, reducing computational costs, and improving the reliability of DR benchmarks. Experiments on 96 real-world datasets validate that MMC and PDS enhance DR benchmark and optimization processes' accuracy and efficiency.


This repository provides:
1. Implementation of the two structural reducibility metrics: PDS and MNC
2. Codes for reproducing the experiments in the related academic paper


## Structrual Reducibility Metrics

The implementation of structrual reducibility metrics (PDS, MNC) are provided under the directory `/src/reducibility/`. MNC and PDS is defined wihtin `pds.py` and `mnc.py`, respectively. 

### Requirements

- Python 3.8+
- Numpy
- Scipy
- Numba

The requirements can be automatically installed by creating the virtual envrionments (e.g., `conda`) and installing the main requirements:
```bash
conda create -n reducibility python==3.9.0
conda activate reducibility
pip install -r Requirements.txt
```

### Specification

#### PDS (Pairwise Distance Shift)

The `pairwise_distance_shift` function computes a complexity metric targeting the global structure of high-dimensional data. It evaluates the shift in pairwise distances within the dataset by analyzing the distribution of distances.

```python
def pairwise_distance_shift(data: np.ndarray) -> float:
```

- Parameters
  - `data (numpy array)`: A numpy array of shape `(n_samples, n_features)` representing the high-dimensional data.
- Returns
  - `float`: The pairwise distance shift value.


> **Example**
> ```python
> from pairwise_distance_shift import pairwise_distance_shift
> import numpy as np
>
> # Example data
>data = np.random.rand(100, 5)
>
> # Compute pairwise distance shift
> shift_value = pairwise_distance_shift(data)
> print(f"Pairwise Distance Shift: {shift_value}")
> ```


#### MNC (Mutual Neighbor Consistency)

The `mutual_neighbor_consistency` function computes a complexity metric targeting the local structure of high-dimensional data. It evaluates the consistency of mutual neighbors within the dataset using k-nearest neighbors (kNN) and shared nearest neighbors (SNN) concepts.

```python
def mutual_neighbor_consistency(data: np.ndarray, k: int) -> float:
```

- Parameters
	- `data (numpy array)`: A numpy array of shape `(n_samples, n_features)` representing the high-dimensional data.
	- `k (int)`: The number of neighbors to consider.

- Returns
	- `float`: The mutual neighbor consistency value.

> **Example**
> ```python
> from mutual_neighbor_consistency import mutual_neighbor_consistency
> import numpy as np
>
> # Example data
> data = np.random.rand(100, 5)
>
> # Compute mutual neighbor consistency
> consistency_value = mutual_neighbor_consistency(data, k=5)
> print(f"Mutual Neighbor Consistency: {consistency_value}")
> ```


## Reproducing the Experiments

The followings are the steps to reproduce the experiments in the paper.
The experiments produces the raw data file that are used to generate the tables and figures in the paper. 


### Setup 

To set up the repository, follow these steps:

1. Download and cleanup datasets
	- Already provided in the repository (for the review process). 

2. Create two virtual environments:
	- For the main requirements:
		```bash
		conda create -n reducibility-main python==3.9.0
		conda activate reducibility-main
		pip install -r requirements.txt
		```

	- For the auto-sklearn requirements (needs separated environment due to dependency issue in auto-sklearn):
		```bash
		conda create -n reducibility-autosklearn python==3.9.18
		conda activate reducibility-autosklearn
		pip install -r requirements-autosklearn.txt
		```



conda create -n reducibility-autosklearn-3 -c conda-forge python==3.8.0 auto-sklearn numpy bayesian-optimization umap-learn

### Main Experiments

Run every codes under the `src` directory. The experiments are divided into three parts: approximation of ground truth structural reducibility, correlation/efficiency analysis, and use cases. 

#### 1. Approximation of the Ground Truth Structural Reducibility of Datasets
Generate the ground truth by navigating to the `ground_truth` directory and running the script:
```bash
conda activate reducibility-main
python3 -m ground_truth._ground_truth
```
See `src/ground_truth/result` directory to check whether the files are generated. Here, the approximated ground truth of each dataset computed by each DR evaluation metrics (`tnc`, `mrre`, `l-tnc`, `srho`, `pr`) can be obtained by finding the maximum `score` of each dataset achievable across all DR techniques.

#### 2. Correlation Analysis

Evaluate the correlation of PDS and MNC scores with ground truth structural reducibility. This result is needed for further experiments and applications. 

**(Step 1)** The evaluation starts by running the following script to apply the reducibility metrics and baselines to the datasets:
```bash
python3  -m exp.01_run_metrics.py
```
Ensure the `src/exp/scores/` directory and its subdirectories are properly generated.


**(Step 2)** Run the script to check how well the reducibility metrics and baselines correlate with the ground truth:
```bash
python3  -m exp.02_accuracy.py
```
Check `exp/results` directory to check whether the `correlation.csv` file is prperly generated.


**(Result and Discussion)**
The below table shows the result of this experiment. Pds and Mnc well explain global and local structural reducibility, respectively, outperforming intrinsic dimensionality metrics. Their ensemble (Pds+Mnc) achieves the best correlation with all types of structural reducibility.
<p align="center">    
<img src="figs/Table1.png" style="width: 50%; height: auto;"/>
</p>


#### 2. Efficinecy Analysis

This evaluation is conducted to evaluate the speed of our structural reducibility metrics.
We reuse the time measurement done while executing the previous experiment. The below script's functionality is to aggregate the time measurements as a single file.

```bash
python3 -m exp.03_time.py
```
Please check the `exp/results` directory to see whether the `time.csv` file is properly generated.

**(Result and Discussion)**
The below figure illustrates the result. As in the figure, Pds and Mnc are slower than the projection-based intrinsic dimensionality metric but are faster than other baselines. 

<p align="center">    <img src="figs/Figure1.png" style="width: 50%; height: auto;"/>
</p>

### Experiments for validating the Applicability of our Metrics on Use Cases

#### Use case 1: Guiding DR Optimization with Structural Reducibility Metrics

The following scripts are used to evaluate the efficiency of our metrics in guiding DR optimization (use case 1)

**(Step 1: Predicting Maximum Accuracy of DR Techniques)** Run the following code to run the script. 

**Warning**: Activate the `reducibility-autosklearn` conda environment to run this.

```bash
conda activate reducibility-autosklearn
python3 -m app.01_01_prediction
```

The script will save the predicted tnc results in the `app/results/predicted_tnc/` directory.

<!-- **(Step 2: Checking te -->

**(Step 2: Enhancing Efficiency of DR Optimization)** 

**Warning**: Switch back to the `reducibility-main` conda environment to run this.

```bash
conda activate reducibility-main
python3 -m app.01_02_optimization
```

The script will produce the results of optimization (time and final accuracy) of each dataset in the `app/optimization/` directory. 

Note that `01_03_aggregate.py` script is used to aggregate the results of the optimization to draw the results as figures using R. This will make `app1_results.csv` file in the `results/` directory.


**(Result and Discussion)**
The results indicate that interrupting iterations based on estimated optimal scores exhibit a substantial reduction in execution time compared to running optimization with a fixed number of iterations.

<p align="center">    <img src="figs/Table2.png" style="width: 50%; height: auto;"/>
</p>
<p align="center">    <img src="figs/Figure2.png" style="width: 50%; height: auto;"/>
</p>


#### Use case 2: Improving the Replicability of DR Benchmarks


The followings are the scripts to evaluate the replicability of DR benchmarks guided by our metrics (use case 2). The first script is used to execute the evaluation, and the second script is used to aggregate the results as a single file.

```bash
python3 -m app.02_01_evaluation 
python3 -m app.02_02_aggregate
```

The script will produce the results of the evaluation in the `app/results/` directory.

**(Result and Discussion)**
Table 3 depicts the results. 

<p align="center">    <img src="figs/Table3.png" style="width: 50%; height: auto;"/>
</p>

