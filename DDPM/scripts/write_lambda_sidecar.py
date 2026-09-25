#!/usr/bin/env python3
"""
Write the per-run lambda-sweep sidecar <run_dir>/rows.json from the metrics
dump produced by pipeline.py (branch ddpm-fid-repro; metrics.json is written
next to the run output, PATH/outputs/<timestamp>/metrics.json).

The sidecar format is identical to the one ablation_regions.py writes on
master, so scripts/plot_lambda_sweep.py can aggregate both.

Usage:
    python scripts/write_lambda_sidecar.py \
        --run_dir <results>/runs/lam1.0_seed0 \
        --lambda 1.0 --seed 0
"""

import argparse
import json
import glob
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="per-run dir (parent of output/ and rows.json)")
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    metrics_paths = glob.glob(os.path.join(args.run_dir, "output", "*", "metrics.json"))
    if not metrics_paths:
        raise SystemExit(f"no metrics.json found under {args.run_dir}/output/*/")
    metrics_path = max(metrics_paths, key=os.path.getmtime)
    metrics = json.load(open(metrics_path))

    def num(k):
        v = metrics.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    row = {
        "experiment": "lambda_sweep",
        "region_mode": "two_corner",
        "interval_mode": "full",
        "lambda": args.lam,
        "seed": args.seed,
        "ua": num("UA"),
        "ta": num("TA"),
        "fid": num("FID"),
    }
    if row["fid"] is None:
        print(f"WARNING: run has no FID (missing ref dataset or fid samples); "
              f"run scripts/fid_backfill_lambda_sweep.py later")

    out = os.path.join(args.run_dir, "rows.json")
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(f"wrote {out}: {row}")


if __name__ == "__main__":
    main()