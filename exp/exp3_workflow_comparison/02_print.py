"""
Print summary of Experiment 3 (workflow comparison)
---------------------------------------------------
For each DR evaluation metric this script prints:

  • Average ± std of projection scores
      - Top‑1 workflow
      - Top‑3 workflow
      - Ground‑truth (conventional)

  • Average ± std of optimization time (seconds)
      - Top‑1 workflow
      - Top‑3 workflow
      - Ground‑truth (conventional)

Directory layout assumed:
./exp/exp3_workflow_comparison/results/<dr_metric_id>/
    ├─ top1_scores.json   # list[dict] : [{<tech>: score}, ...]
    ├─ top1_times.json    # list[float]
    ├─ top3_scores.json   # list[dict] : [{<tech1>: score, <tech2>: score, ...}, ...]
    ├─ top3_times.json    # list[float]
    ├─ gt_scores.json     # list[float]
    └─ gt_times.json      # list[float]
"""

import json
import numpy as np
from pathlib import Path
import exp.load_config as lc
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("./exp/exp3_workflow_comparison/results")
DR_METRICS = lc.load_config("METRICS")   # list of dicts with "id" key


def load_json(path, default=None):
    """Safely load JSON and return default on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def stats(arr):
    """Return mean and std as formatted strings."""
    if not arr:
        return "nan ± nan"
    mean = np.mean(arr)
    std  = np.std(arr)
    return f"{mean:.4f} ± {std:.4f}"


def extract_top_scores(score_dicts):
    """
    From a list of dicts (each dict = {tech: score, ...}),
    keep only the best score per dict.
    """
    return [max(d.values()) if d else np.nan for d in score_dicts]


def main():
    if not BASE_DIR.exists():
        print("Base directory not found:", BASE_DIR)
        return

    print("#" * 90)
    print("WORKFLOW COMPARISON SUMMARY (Experiment 3)")
    print("#" * 90)

    for dr in DR_METRICS:
        dr_id = dr["id"]
        dr_dir = BASE_DIR / dr_id
        if not dr_dir.exists():
            continue

        # ---- Load data ----
        top1_scores_raw = load_json(dr_dir / "top1_scores.json", [])
        top1_times      = load_json(dr_dir / "top1_times.json",  [])
        top3_scores_raw = load_json(dr_dir / "top3_scores.json", [])
        top3_times      = load_json(dr_dir / "top3_times.json",  [])
        gt_scores       = load_json(dr_dir / "gt_scores.json",   [])
        gt_times        = load_json(dr_dir / "gt_times.json",    [])

        # Convert dict‑lists to list[float]
        top1_scores = [ list(d.values())[0] if d else np.nan for d in top1_scores_raw ]
        top3_scores = extract_top_scores(top3_scores_raw)

        # ---- Print summary ----
        print(f"\nDR Metric : {dr_id}")
        print("  Scores")
        print("    Top‑1 :", stats(top1_scores))
        print("    Top‑3 :", stats(top3_scores))
        print("    GT    :", stats(gt_scores))

        print("  Times (s)")
        print("    Top‑1 :", stats(top1_times))
        print("    Top‑3 :", stats(top3_times))
        print("    GT    :", stats(gt_times))


if __name__ == "__main__":
    main()
