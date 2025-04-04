<p align="center">
  <h2 align="center">Structural Complexity Metrics</h2>
	<p align="center">Repository for the paper <i>Dataset-Adaptive Dimensionality Reduction</i></p>
</p>

---

We introduce two structural complexity metrics, Pairwise Distance Shift (PDS) and Mutual Neighbor Consistency (MNC), which are designed to quantify the maximum achievable structural consistency between high-dimensional (HD) datasets and their 2D projections. These metrics guide dimensionality reduction (DR) by predicting optimal iteration numbers, reducing computational costs, and improving the reliability of DR benchmarks. Experiments on 96 real-world datasets validate that MMC and PDS enhance DR benchmark and optimization processes' accuracy and efficiency.


This repository provides:
1. Implementation of the two structural complexity metrics: PDS and MNC
2. Codes for reproducing the experiments in the related academic paper

### Requirements

- Python 3.8+
- Numpy
- Scipy
- Numba

The requirements can be automatically installed by running
```bash
conda create -n complexity python==3.9.0
conda activate complexity
pip install -r requirements.txt
```

# Structrual Complexity Metrics
Located in [`/src/metrics/`](src/metrics):
- `pds.py`: Pairwise Distance Shift (global structure)
- `mnc.py`: Mutual Neighbor Consistency (local structure)

### PDS (Pairwise Distance Shift) Example
``` python
from src.metrics.pds import pairwise_distance_shift
import numpy as np

data = np.random.rand(100, 5)
shift_value = pairwise_distance_shift(data)
print(f"PDS: {shift_value:.3f}")
```


### MNC (Mutual Neighbor Consistency) Example
```python
from src.metrics.mnc import mutual_neighbor_consistency
import numpy as np

data = np.random.rand(100, 5)
consistency_value = mutual_neighbor_consistency(data, k=5)
print(f"MNC: {consistency_value:.3f}")
```

# Workflow

### Pretraining
Trains models that predict the maximum achievable accuracy of DR techniques from complexity scores.
These predictors guide downstream optimization and selection.

```python
from src.pretrain import train_model
import numpy as np

data = np.random.rand(100, 5)
labels = np.random.rand(100)

model = train_model(data, labels, config={"epochs": 10, "lr": 0.001})
```

### Early terminating optimzation
Used to save computation by:

**Step 1 — Selecting Effective DR Techniques**

Predict performance for each technique and skip those with low expected accuracy.

**Step 2 — Early Stopping**

Stop hyperparameter tuning once predicted performance is reached.
```python
from src.opt_early_stop import EarlyStopper

stopper = EarlyStopper(patience=3, delta=0.01)

for epoch in range(100):
    val_score = evaluate_model(...)
    if stopper.should_stop(val_score):
        print("Early stopping triggered.")
        break
```

# Reproducing the Experiments

The following are the steps to reproduce the experiments in the paper.
The experiments produce the raw data file that is used to generate the tables and figures in the paper. 


### Setup 

To set up the repository, follow these steps:

1. Download and cleanup datasets
	- Already provided in the repository (for the review process). 

2. Create two virtual environments:
	- For the main requirements:
		```bash
		conda create -n complexity-main python==3.9.0
		conda activate complexity-main
		pip install -r requirements.txt
		```

	- For the auto-sklearn requirements (needs separated environment due to dependency issue in auto-sklearn):
		```bash
		conda create -n complexity-autosklearn python==3.9.18
		conda activate complexity-autosklearn
		pip install -r requirements-autosklearn.txt
		```

conda create -n complexity-autosklearn-3 -c conda-forge python==3.8.0 auto-sklearn numpy bayesian-optimization umap-learn

## Main Experiments

### Experiment 1 — Structural Complexity Metrics

**Path:** `exp/exp1_structural_metrics`

This experiment evaluates the validity of PDS and MNC as structural complexity metrics by comparing them to ground truth model performance and analyzing their correlation with intrinsic dimensionality.

```bash
cd exp/exp1_structural_metrics
bash run.sh
```

**Subscripts:**
- `01_generate_ground_truth.py`: Train regression models to estimate DR effectiveness
- `02_compute_intrinsic_dim.py`: Compute intrinsic dimensionality metrics
- `03_train.py`: Train models using metrics as features
- `04_evaluate.py`: Evaluate correlation with DR performance

**Results:**
![Table 1: Accuracy of structural complexity metrics](figs/Table1.png)
![Figure 3: Runtime of structural complexity metrics](figs/Figure3.png)

---

### Experiment 2 — Adaptive Workflow

**Path:** `exp/exp2_adaptive_workflow`

Tests whether our structural complexity metrics can support a dataset-adaptive pipeline for model selection and tuning.

```bash
cd exp/exp2_adaptive_workflow
bash run.sh
```

**Subscripts:**
- `01_evaluate_regression_models.py`: Use pretraining to predict best DR methods
- `02_evaluate_predicting_dr.py`: Evaluate ability to guide DR method selection
- `03_evaluate_early_stop.py`: Evaluate optimization efficiency with early stopping

**Results:**
![Table 2: Accuracy of baseline metrics and PDS+MNC](figs/Table2.png)
![Figure 4: Distribution of correlations](figs/figure4.png)
---

### Experiment 3 — Workflow Comparison

**Path:** `exp/exp3_workflow_comparison`

Final experiment that compares the **dataset-adaptive workflow** against a conventional fixed-pipeline workflow.

```bash
cd exp/exp3_workflow_comparison
python evaluate_workflow_improvement.py
```

**Results:**
![Figure 6: Comparison of the performance of three workflows](figs/Figure6.png)
