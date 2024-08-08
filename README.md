<p align="center">
  <h2 align="center">Structural Reducibility Metrics</h2>
	<p align="center">Repository for the paper <i>Measuring the Structural Reducibility of Datasets for Fast and Accurate Dimensionality Reduction</i></p>
</p>

---

We introduce two structural reducibility metrics, Pairwise Distance Shift (PDS) and Mutual Neighbor Consistency (MNC), which are designed to quantify the maximum achievable structural consistency between high-dimensional (HD) datasets and their 2D projections. These metrics guide dimensionality reduction (DR) by predicting optimal iteration numbers, reducing computational costs, and improving the reliability of DR benchmarks. Experiments on 96 real-world datasets validate that MMC and PDS enhance DR benchmark and optimization processes' accuracy and efficiency.


This repository provides:
1. Implementation of the two structural reducibility metrics: PDS and MNC
2. Codes for reproducding the experiments in the related academic paper


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


# Setup

To set up the repository, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/hj-n/labeled-datasets.git
    ```

2. Navigate to the `labeled-datasets` directory and remove the `npy` zip file:
    ```bash
    cd labeled-datasets
    rm *.zip
    ```

3. Create two virtual environments:
    - For the main requirements:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install -r Requirements.txt
        ```

    - For the auto-sklearn requirements:
        ```bash
        python3 -m venv venv-autosklearn
        source venv-autosklearn/bin/activate
        pip install -r requirements-autosklearn.txt
        ```

# Usage

## 1. Ground Truth Generation
Generate the ground truth by navigating to the `ground_truth` directory and running the script:
```bash
source ../venv/bin/activate
cd src/ground_truth
python3 _ground_truth.py
```
Check that the `src/ground_truth/result` directory is generated.

## 2. Experiments
Run the experiments in the following order:

### Experiment 01
Evaluate the correlation of PDS and MNC scores with ground truth structural reducibility. This result is needed for further experiments and applications.
```bash
python3 exp/01_run_metrics.py
```
Ensure the `src/exp/scores/` directory is generated.

### Experiment 02: Correlation Analysis
Evaluate how well our reducibility metrics and baselines predict surrogate ground truths. Test the ensemble of PDS and MNC.
```bash
python3 exp/02_accuracy.py
```

#### Result and Discussion
The below table shows the result of this experiment. Pds and Mnc well explain global and local structural reducibility, respectively, outperforming intrinsic dimensionality metrics. Their ensemble (Pds+Mnc) achieves the best correlation with all types of structural reducibility.
<p align="center">    
<img src="figs/Table1.png" style="width: 50%; height: auto;"/>
</p>

### Experiment 03: Efficiency Analysis
Evaluate the speed of our structural reducibility metrics.
```bash
python3 exp/03_time.py
```

#### Result and Discussion
The below figure illustrates the result. As in the figure, Pds and Mnc are slower than the projection-based intrinsic dimensionality metric but are faster than other baselines. 

<p align="center">    <img src="figs/Figure1.png" style="width: 50%; height: auto;"/>
</p>



## 3. Applications
### Application 01: Predicting Maximum Accuracy of DR Techniques
**Warning**: Activate the `venv-autosklearn` environment to run this.

- Predicting Maximum Accuracy of DR Techniques:
```bash
python3 app/02_01_prediction.py 
```
- Enhancing Efficiency of DR Optimization:
```bash
python3 app/02_02_optimization.py
```
- Test and Convert to DataFrame:
```bash
python3 app/02_03_test_and_to_df.py
```

#### Result and Discussion
The results indicate that interrupting iterations based on estimated optimal scores exhibit a substantial reduction in execution time compared to running optimization with a fixed number of iterations.

<p align="center">    <img src="figs/Table2.png" style="width: 50%; height: auto;"/>
</p>
<p align="center">    <img src="figs/Figure2.png" style="width: 50%; height: auto;"/>
</p>

### Application 02
Improve the replicability of DR benchmarks:
```bash
python3 app/03_01_evaluation.py
python3 app/03_02_cleanup.py
```

#### Result and Discussion
Table 3 depicts the results. +global 916 & −local, −local, and +global improve the replicability of the 917 DR benchmark using local evaluation metrics (T&C, MRREs).

<p align="center">    <img src="figs/Table3.png" style="width: 50%; height: auto;"/>
</p>

## 4. Deactivate the Virtual Environment
```bash
deactivate
```
