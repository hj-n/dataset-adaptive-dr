"""
Print Experiment 1 results
--------------------------
Evaluation A. Runtime
  • Average ± std of execution time for each complexity metric
    (mnc, pds, pdsmnc, geometric_intdim, projection_intdim, dr_ensemble)

Evaluation B. Predictive Power
  • R² values of each complexity metric (competitor) when predicting
    ground‑truth structural complexity produced by every DR evaluation metric,
    for five regression models (linear, polynomial, knn, rf, gb).

Directory layout assumed
------------------------
exp/exp1_metrics/results/
    ├─ metrics/
    │    ├─ mnc_25.json , mnc_50.json , mnc_75.json
    │    ├─ pds.json
    │    ├─ geometric_intdim.json
    │    └─ projection_intdim.json
    ├─ ground_truth/<dr_tech>/<dr_metric_id>/<dataset>.json   # for dr_ensemble time
    └─ final/
         ├─ runtime.csv        # (written by previous code; not used here)
         └─ correlations.csv   # columns: dr_metric, competitor, regression_model, r2
"""

import json
import csv
import math
from pathlib import Path
import numpy as np
import exp.load_config as lc
import src.modules.load as l
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Config & helper functions
# ------------------------------------------------------------------------------
BASE_METRIC_DIR = Path("exp/exp1_metrics/results/metrics")
GROUND_DIR      = Path("exp/exp1_metrics/results/ground_truth")
CORR_CSV        = Path("exp/exp1_metrics/results/final/correlations.csv")

DR_TECHNIQUES = lc.load_config("DR")
DR_METRICS    = lc.load_config("METRICS")          # list of dicts with "id"
DATASET_PATH  = lc.load_config("DATASET_PATH")
DATASET_LIST  = l.load_names(DATASET_PATH)

REGRESSION_MODELS = {
    "linear":      "Linear Regression",
    "polynomial":  "Polynomial Regression",
    "knn":         "KNN",
    "rf":          "Random Forest",
    "gb":          "Gradient Boosting"
}

# Complexity metrics we handle
METRICS_TIME = {
    "mnc":                ["mnc_25.json"],                       # single file
    "pds":                ["pds.json"],
    "pdsmnc":             ["mnc_25.json", "mnc_50.json", "mnc_75.json", "pds.json"],
    "geometric_intdim":   ["geometric_intdim.json"],
    "projection_intdim":  ["projection_intdim.json"],
}

# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------
def load_json(path):
    """Return JSON content or None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def stats(arr):
    """Return mean ± std formatted; arr is list[float]."""
    if not arr:
        return "nan ± nan"
    return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

# ------------------------------------------------------------------------------
# Evaluation A – Runtime
# ------------------------------------------------------------------------------
def compute_runtime():
    """Return dict {metric: [times per dataset]} and dr_ensemble times list."""
    runtimes = {m: [] for m in METRICS_TIME}
    dr_ensemble_time = []

    # Pre‑load JSONs for speed
    cache = {}
    for files in METRICS_TIME.values():
        for fname in files:
            if fname not in cache:
                cache[fname] = load_json(BASE_METRIC_DIR / fname)

    for ds in DATASET_LIST:
        # Each complexity metric
        for metric, files in METRICS_TIME.items():
            total = 0.0
            for fname in files:
                json_obj = cache.get(fname, {})
                total += json_obj.get(ds, {}).get("time", math.nan)
            runtimes[metric].append(total)

        # dr_ensemble time (sum of all DR techniques for the first DR_METRIC only)
        dr_metric_id = DR_METRICS[0]["id"]
        total_t = 0.0
        for tech in DR_TECHNIQUES:
            gt_path = GROUND_DIR / tech / dr_metric_id / f"{ds}.json"
            obj = load_json(gt_path) or {}
            total_t += obj.get("time", math.nan)
        dr_ensemble_time.append(total_t)

    runtimes["dr_ensemble"] = dr_ensemble_time
    return runtimes

# ------------------------------------------------------------------------------
# Evaluation B – Predictive Power
# ------------------------------------------------------------------------------
def load_correlations():
    """
    Parse correlations.csv into nested dict:
      corr[dr_metric_id][competitor][reg_model] = r2_value
    """
    corr = {}
    if not CORR_CSV.exists():
        return corr

    with open(CORR_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dr_m   = row["dr_metric"]
            comp   = row["competitor"]
            reg    = row["regression_model"]
            r2     = float(row["r2"])
            corr.setdefault(dr_m, {}).setdefault(comp, {})[reg] = r2
    return corr

# ------------------------------------------------------------------------------
# Main printing
# ------------------------------------------------------------------------------
def main():
    # ===== Evaluation A =====
    print("#" * 90)
    print("EVALUATION A – Runtime")
    print("#" * 90)

    runtimes = compute_runtime()
    for metric, times in runtimes.items():
        print(f"{metric:<20}: {stats(times)}")

    # ===== Evaluation B =====
    print("\n" + "#" * 90)
    print("EVALUATION B – Predictive Power (R²)")
    print("#" * 90)

    corr = load_correlations()
    if not corr:
        print("correlations.csv not found or empty.")
        return

    # Loop over ground‑truth DR metrics
    for dr in DR_METRICS:
        dr_id = dr["id"]
        print(f"\nGround truth based on DR metric: {dr_id}")

        # Loop over regression models
        for reg_key, reg_name in REGRESSION_MODELS.items():
            print(f"  {reg_name}:")
            # Loop over competitors (metrics) – sorted for consistency
            for comp in sorted(corr.get(dr_id, {})):
                r2 = corr[dr_id][comp].get(reg_key, float("nan"))
                val = f"{r2:.4f}" if not math.isnan(r2) else "nan"
                print(f"    {comp:<20} {val}")
    print()

if __name__ == "__main__":
    main()
