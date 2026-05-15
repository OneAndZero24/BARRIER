#!/usr/bin/env python3
"""Compute per-remain-fraction average metrics from W&B runs.

Example:
  python scripts/average_remain_fraction_metrics.py \
    --entity oneandzero24 \
    --project intact-ddpm \
    --group cifar10-ablate-remain-fraction
"""

import argparse
import math
import statistics
from collections import defaultdict

import wandb


def parse_args():
    p = argparse.ArgumentParser(description="Average DDPM remain-fraction ablation metrics from W&B")
    p.add_argument("--entity", required=True, help="W&B entity (team or username)")
    p.add_argument("--project", required=True, help="W&B project")
    p.add_argument("--group", required=True, help="W&B run group")
    p.add_argument("--metric", action="append", default=[], help="Metric key to aggregate; can be passed multiple times")
    p.add_argument("--include-running", action="store_true", help="Include runs not in finished state")
    return p.parse_args()


def main():
    args = parse_args()
    api = wandb.Api()

    default_metrics = ["FID", "UA", "TA", "inception_score", "sfid", "precision", "recall"]
    metrics = args.metric if args.metric else default_metrics

    path = f"{args.entity}/{args.project}"
    runs = api.runs(path=path, filters={"group": args.group})

    by_frac = defaultdict(list)
    skipped = 0

    for run in runs:
        if (not args.include_running) and run.state != "finished":
            continue

        cfg = run.config or {}
        intact = cfg.get("intact", {}) if isinstance(cfg, dict) else {}
        remain_fraction = intact.get("remain_fraction")
        if remain_fraction is None:
            skipped += 1
            continue

        row = {"run": run.name, "state": run.state}
        for m in metrics:
            val = run.summary.get(m)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                row[m] = float(val)

        by_frac[float(remain_fraction)].append(row)

    if not by_frac:
        print("No runs found for the requested filters.")
        return

    print(f"Runs grouped by remain_fraction for {path} / group={args.group}")
    if skipped:
        print(f"Skipped {skipped} runs without intact.remain_fraction in config.")
    print()

    header = ["remain_fraction", "n"]
    for m in metrics:
        header.extend([f"{m}_mean", f"{m}_std"])
    print("\t".join(header))

    for frac in sorted(by_frac.keys(), reverse=True):
        rows = by_frac[frac]
        out = [f"{frac}", str(len(rows))]

        for m in metrics:
            vals = [r[m] for r in rows if m in r]
            if not vals:
                out.extend(["NA", "NA"])
                continue

            mu = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out.extend([f"{mu:.6f}", f"{sd:.6f}"])

        print("\t".join(out))


if __name__ == "__main__":
    main()
