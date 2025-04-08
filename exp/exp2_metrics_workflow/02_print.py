"""
Print Experiment 2 results with evaluation loop outermost
--------------------------------------------------------
Loop order:   Evaluation  →  Complexity Metric  →  DR Metric

Evaluation 1 : exp1_correlations.json  (R² per DR technique)
Evaluation 2 : exp2_top_1_accuracy.json , exp2_top_3_accuracy.json
Evaluation 3 : exp3_opt_time.json , exp3_gt_time.json ,
               exp3_opt_score.json , exp3_gt_score.json
"""

import json
import math
from pathlib import Path

BASE_DIR = Path("./exp/exp2_metrics_workflow/results/prediction")
COMPLEXITY_METRICS = ["pdsmnc", "intdim_proj", "intdim_geo"]

def load_json(path, default=None):
    """Safely load a JSON file and return default on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def fmt(val):
    """Format float/int nicely; return 'nan' for invalid numbers."""
    return f"{val:.4f}" if isinstance(val, (float, int)) and not math.isnan(val) else "nan"

def print_eval1(cm_dir, dr_id):
    """Print Evaluation 1 results for a given directory."""
    data = load_json(cm_dir / dr_id / "exp1_correlations.json", {})
    if not data:
        print("    (no data)")
        return
    for tech, r2 in data.items():
        print(f"    {tech:<15} : {fmt(r2)}")

def print_eval2(cm_dir, dr_id):
    """Print Evaluation 2 results for a given directory."""
    top1 = load_json(cm_dir / dr_id / "exp2_top_1_accuracy.json", float("nan"))
    top3 = load_json(cm_dir / dr_id / "exp2_top_3_accuracy.json", float("nan"))
    print(f"    Top‑1 accuracy : {fmt(top1)}")
    print(f"    Top‑3 accuracy : {fmt(top3)}")

def print_eval3(cm_dir, dr_id):
    """Print Evaluation 3 results for a given directory."""
    opt_t = load_json(cm_dir / dr_id / "exp3_opt_time.json",  float("nan"))
    gt_t  = load_json(cm_dir / dr_id / "exp3_gt_time.json",   float("nan"))
    opt_s = load_json(cm_dir / dr_id / "exp3_opt_score.json", float("nan"))
    gt_s  = load_json(cm_dir / dr_id / "exp3_gt_score.json",  float("nan"))

    speedup = opt_t / gt_t if gt_t and not math.isnan(gt_t) else float("nan")
    q_ratio = opt_s / gt_s if gt_s and not math.isnan(gt_s) else float("nan")

    print(f"    opt_time       : {fmt(opt_t)}")
    print(f"    gt_time        : {fmt(gt_t)}")
    print(f"    speedup(opt/gt): {fmt(speedup)}")
    print(f"    opt_score      : {fmt(opt_s)}")
    print(f"    gt_score       : {fmt(gt_s)}")
    print(f"    quality(opt/gt): {fmt(q_ratio)}")

def main():
    if not BASE_DIR.exists():
        print("Base directory not found:", BASE_DIR)
        return

    # Evaluation loop outermost
    for eval_id, eval_fn in [("Evaluation 1", print_eval1),
                             ("Evaluation 2", print_eval2),
                             ("Evaluation 3", print_eval3)]:

        print("\n" + "#" * 90)
        print(f"{eval_id.upper()}")
        print("#" * 90)

        # Loop over complexity metrics
        for cm in COMPLEXITY_METRICS:
            cm_dir = BASE_DIR / cm
            if not cm_dir.exists():
                continue

            print(f"\n== Complexity Metric: {cm} ==")

            # Loop over DR metric folders
            for dr_dir in sorted([d for d in cm_dir.iterdir() if d.is_dir()]):
                dr_id = dr_dir.name
                print(f"\n  -- DR Metric: {dr_id} --")
                eval_fn(cm_dir, dr_id)

if __name__ == "__main__":
    main()
