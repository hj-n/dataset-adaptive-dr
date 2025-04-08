<p align="center">
  <h2 align="center">Dataset-Adaptive Workflow for Optimizing Dimensionality Reduction</h2>
	<p align="center">Repository for the paper <i>Dataset-Adaptive Dimensionality Reduction</i></p>
</p>

---

We introduce the *Dataset-Adaptive workflow* for optimizing dimensionality reduction (DR) techniques, which improves the efficiency of the optimization without compromising the accuracy. The workflow is based on two *structural complexity metrics*, Pairwise Distance Shift (PDS) and Mutual Neighbor Consistency (MNC). Our approach is built upon the previous findings that certain patterns are more prominent in HD data. Based on this finding, our approach first quantifies the prominence of these patterns to estimate the difficulty of accurately projecting the data into lower-dimensional spaces. We introduce structural complexity metrics to measure these patterns and use these scores to predict the maximum accuracy achievable by DR techniques. The metrics thus enhance the efficiency of DR optimization by (1) guiding the selection of an appropriate DR technique for a given dataset and (2) enabling early termination of optimization once near-optimal hyperparameters have been reached, avoiding unnecessary computations.

In this repository, we provide the implementation of two structural complexity metrics and the dataset-adaptive workflow. We also provide the code to reproduce the experiments in our paper.



### Dataset-Adaptive workflow

The `/src` directory contains the implementation of the dataset-adaptive workflow. First, `DatasetAdaptiveDR.py` orchestrates the entire pipeline, exposing a high‑level API for adding datasets, training prediction models, and recommending optimal DR techniques and hyperparameters.

- **`metrics/`** – complexity metric implementations:
  - `pds.py`: computes **Pairwise Distance Shift (PDS)**, a global‑structure metric.
  - `mnc.py`: computes **Mutual Neighbor Consistency (MNC)**, a local‑structure metric.
  - `helpers/`: utility code (fast k‑NN/SNN) shared by the metrics.
- **`intrinsic_dim/`** – optional intrinsic‑dimensionality estimators (`geometric.py`, `projection.py`) that can serve as alternative complexity metrics.
- **`modules/`** – workflow components:
  - `opt_conv.py`: unified hyperparameter‑optimization wrapper with early termination.
  - `train.py`: regression‑model training routine mapping complexity metrics to maximum achievable accuracy.

This modular layout separates metric computation, optimization logic, and high‑level orchestration, making the library easy to extend and maintain.

### Requirements



Python 3.9.5 or higher (we recommend toe use Python 3.9.5 for reproducing the experiments).

The requirements can be automatically installed by running:
```bash
conda create --name complexity --file python==3.9.5 
```

### API Reference

#### `DatasetAdaptiveDR`

High‑level orchestrator that learns to predict the maximum achievable accuracy of a DR technique from dataset‑level complexity metrics and then performs guided hyper‑parameter optimization.

```python
DatasetAdaptiveDR(
    dr_techniques: List[str],
    dr_metric: str,
    dr_metric_names: List[str],
    params: Dict[str, Any],
    is_higher_better: bool,
    init_points: int,
    n_iter: int,
    complexity_metric: str,
    training_info: Dict[str, Any] = {
        "single_task_time": 30,
        "total_task_time": 600,
        "memory_limit": 10000,
        "cv_fold": 5,
    },
)
```

| Argument            | Type        | Description                                                                                                                    |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `dr_techniques`     | list of str | Subset of supported DR algorithms (`"pca"`, `"umap"`, `"tsne"`, `"umato"`, `"lle"`, `"isomap"`).                               |
| `dr_metric`         | str         | Evaluation metric used during opti‑ization (`"tn` wrapper with early termination.`c"`, `"mrre"`, `"l_tnc"`, `"srho"`, `"pr"`). |
| `dr_metric_names`   | list of str | Names of the metrics to report for each DR technique.                                                                          |
| `params`            | dict        | Search space definition for each DR technique.                                                                                 |
| `is_higher_better`  | bool        | Whether a larger metric value indicates better quality.                                                                        |
| `init_points`       | int         | Initial random trials before Bayesian optimization.                                                                            |
| `n_iter`            | int         | Number of optimization iterations.                                                                                             |
| `complexity_metric` | str         | Complexity metric to use (`"mnc"`, `"pds"`, `"pdsmnc"`, `"in`Dataset-Adaptive Workflow`tdim_proj"`, `"intdim_geo"`).           |
| `training_info`     | dict        | Runtime / memory budget and cross‑validation settings for the internal learner.                                                |

**Key methods**

| Method                                      | Returns            | Purpose                                                                                                                            |
| ------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `add_dataset(data, labels=None)`            | –                  | Adds a dataset, computes its complexity score(s) and its maximum achievable accuracy per DR technique.                             |
| `fit()`                                     | –                  | Trains a regression model (one per DR technique) that maps complexity metrics to accuracy.                                         |
| `predict(data)`                             | `Dict[str, float]` | Predicts the achievable accuracy of each DR technique on an unseen dataset.                                                        |
| `predict_opt(data, top_num=1, labels=None)` | `(scores, params)` | Performs guided hyper‑parameter search on the best‑predicted `top_num` techniques and returns their optimized scores & parameters. |

---

#### `mutual_neighbor_consistency` (`mnc`)

Local‑structure complexity metric measuring how consistently *k*‑nearest‑neighbor relationships are preserved among shared nearest neighbors.

```python
a = mutual_neighbor_consistency(data: np.ndarray, k: int) -> float
```

| Parameter | Type                                     | Description                              |
| --------- | ---------------------------------------- | ---------------------------------------- |
| `data`    | `np.ndarray` *(n\_samples, n\_features)* | High‑dimensional dataset.                |
| `k`       | int                                      | Number of nearest neighbors to consider. |

Returns a single scalar between 0 and 1 (higher means more consistent neighborhood structure, i.e., *easier* for DR).

---

#### `pairwise_distance_shift` (`pds`)

Global‑structure complexity metric capturing how much the distribution of pairwise distances deviates from uniformity.

```python
s = pairwise_distance_shift(data: np.ndarray) -> float
```

| Parameter | Type                                     | Description               |
| --------- | ---------------------------------------- | ------------------------- |
| `data`    | `np.ndarray` *(n\_samples, n\_features)* | High‑dimensional dataset. |

Returns a scalar; larger values indicate more homogeneous distance distribution (hence *harder* for DR).



---

### Reproducing Experiment 1: Validating the Structural Complexity Metrics

The **`exp/exp1_metrics`** folder contains the full pipeline for Experiment 1 in the paper: assessing whether PDS and MNC (and their combinations) are reliable predictors of dataset reducibility.

> **Quick start** – assuming your environment is activated and `python -m pip install -r requirements.txt` (or the `conda` environment above) has been executed:
>
> ```bash
> # 1 – Generate ground‑truth maximum accuracy for every dataset / DR technique / DR metric
> python3 -m exp.exp1_metrics.01_generate_ground_truth
>
> # 2 – Compute structural‑complexity metrics (MNC, PDS, intrinsic‑dim)
> python3 -m exp.exp1_metrics.02_compute_metrics
>
> # 3 – Train regression models that map complexity metrics → ground truth, saving R² scores
> python3 -m exp.exp1_metrics.03_train
>
> # 4 – Print summary tables (scalability + accuracy)
> python3 -m exp.exp1_metrics.04_print
> ```
>
> **Tip** : `02_compute_metrics.py` selects a CUDA device via `cuda.select_device(1)`. Change the index to match your GPU setup, or comment the line out for CPU execution.

#### Configuration

All scripts read a single configuration file via `exp/load_config.py`. Edit `exp/config.json` (or the file your fork uses) to point to your datasets and tweak hyper‑parameters:

| Key | Purpose |
| --- | ------- |
| `DATASET_PATH` | Root folder containing raw datasets (each in its own sub‑folder). |
| `MAX_POINTS` | Maximum number of points to sample from each dataset for faster experiments. |
| `DR` | List of DR techniques to evaluate (`["pca", "umap", ...]`). |
| `METRICS` | List of DR evaluation metrics, each with `id`, `names`, `params`, `is_higher_better`. |
| `INIT_POINTS`, `MAX_ITER` | Bayesian‑optimization budget for ground‑truth search. |
| `REGRESSION_ITER` | Number of shuffles / restarts when cross‑validating regression models. |

#### Expected output structure

After the four scripts complete, the directory tree will look like:

```
exp/exp1_metrics/results/
 ├─ ground_truth/<dr_tech>/<dr_metric>/<dataset>.json   # step 1
 ├─ metrics/                                           # step 2
 │    ├─ mnc_25.json  mnc_50.json  mnc_75.json
 │    ├─ pds.json
 │    ├─ geometric_intdim.json
 │    └─ projection_intdim.json
 └─ final/                                             # step 3
      ├─ correlations.csv   # R² per (ground‑truth metric × competitor × regressor)
      └─ runtime.csv        # (optional, if you store runtimes separately)
```

Running **`04_print.py`** will summarise:

* **Scalability** – mean ± std runtime for each complexity metric and the exhaustive DR‑ensemble baseline.
* * **Accuracy** – R² scores showing how well each complexity metric (competitor) predicts ground‑truth structural complexity under five regression models (linear, polynomial, KNN, random forest, gradient boosting).

Feel free to adapt the configuration or extend the scripts to reproduce additional experiments reported in the paper.


---

### Reproducing Experiment 2: Assessing the Dataset‑Adaptive Workflow

The **`exp/exp2_metrics_workflow`** folder evaluates whether the structural‑complexity metrics are *useful* inside the full Dataset‑Adaptive workflow.

> **Quick start**
>
> ```bash
> # 1 – Run the end‑to‑end evaluation (training, prediction, early‑stopping analysis)
> python3 -m exp.exp2_metrics_workflow.01_evaluate
>
> # 2 – Print consolidated results (accuracy, recommendation quality, scalability)
> python3 -m exp.exp2_metrics_workflow.02_print
> ```
>
> The script will iterate over three complexity metric and baseline configurations (`pdsmnc`, `intdim_proj`, `intdim_geo`) and every DR evaluation metric defined in your `exp/config.json`.

#### What is measured?

| Aspect | Output file(s) | Meaning |
| ------ | -------------- | ------- |
| **Predictive Power** | `exp1_correlations.json` | R² between predicted and ground‑truth maximum accuracy for each DR technique. |
| **Recommendation Quality** | `exp2_top_1_accuracy.json`, `exp2_top_3_accuracy.json` | Fraction of test datasets where the true best (top‑1) or any of the true top‑3 DR techniques appear in the model’s recommendations. |
| **Effectiveness of Early Termination** | `exp3_opt_time.json`, `exp3_gt_time.json`, `exp3_opt_score.json`, `exp3_gt_score.json` | Runtime speed‑up achieved by early termination (opt/gt) and the corresponding quality ratio (opt_score/gt_score). |

#### Expected output structure

```
exp/exp2_metrics_workflow/results/prediction/
  ├─ <complexity_metric>/
  │    └─ <dr_metric_id>/
  │         ├─ exp1_correlations.json   # Accuracy
  │         ├─ exp2_top_1_accuracy.json # Recommendation Quality
  │         ├─ exp2_top_3_accuracy.json
  │         ├─ exp3_opt_time.json      # Scalability
  │         ├─ exp3_gt_time.json
  │         ├─ exp3_opt_score.json
  │         └─ exp3_gt_score.json
```

Running **`02_print.py`** will summarise all three aspects for every (complexity‑metric × DR‑metric) pair in a concise console report.

Adjust the configuration parameters (`TRAINING_SIZE`, `TEST_SIZE`, `TRAINING_INFO`, etc.) in `exp/config.json` to match your computational budget or experimental goals.


---

### Reproducing Experiment 3: Comparing the Dataset‑Adaptive and Conventional Workflows

The **`exp/exp3_workflow_comparison`** folder asks a holistic question: *Does the Dataset‑Adaptive workflow actually outperform the conventional brute‑force approach in practice?*

> **Quick start**
>
> ```bash
> # 1 – Evaluate the Dataset‑Adaptive workflow (Top‑1 / Top‑3 recommendations from Dataset-adaptive DR) against the ground‑truth baseline
> python3 -m exp.exp3_workflow_comparison.01_evaluate
>
> # 2 – Print aggregated statistics (accuracy + scalability)
> python3 -m exp.exp3_workflow_comparison.02_print
> ```
>
> The script fixes the complexity‑metric configuration to **`pdsmnc`**—the best performer from Experiment 2—and iterates over every DR evaluation metric defined in `exp/config.json`.

#### What is measured?

| Aspect | Metric | Output file(s) | Meaning |
| ------ | ------ | -------------- | ------- |
| **Accuracy** | Projection score | `top1_scores.json`, `top3_scores.json`, `gt_scores.json` | Best score obtained by the Top‑1 / Top‑3 Dataset‑Adaptive recommendations vs. the exhaustive ground‑truth search. |
| **Scalability** | Optimization time (s) | `top1_times.json`, `top3_times.json`, `gt_times.json` | Wall‑clock time required to reach the above scores. |

#### Expected output structure

```
exp/exp3_workflow_comparison/results/
  └─ <dr_metric_id>/
       ├─ top1_scores.json   # list[dict]
       ├─ top1_times.json    # list[float]
       ├─ top3_scores.json   # list[dict]
       ├─ top3_times.json    # list[float]
       ├─ gt_scores.json     # list[float]
       └─ gt_times.json      # list[float]
```

Running **`02_print.py`** summarises, for each DR evaluation metric, the mean ± std of projection scores and optimization times across datasets:

* **Top‑1 workflow** – using only the single highest‑ranked technique predicted by the model.
* **Top‑3 workflow** – taking the best score among the three techniques predicted by the model.
* **Ground‑truth (conventional)** – exhaustive optimization over *all* DR techniques.

These numbers reveal both the quality retained by the Dataset‑Adaptive recommendations and the speed‑up they provide over brute‑force search.

