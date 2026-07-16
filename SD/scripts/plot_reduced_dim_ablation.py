#!/usr/bin/env python3
"""Aggregate metrics from reduced_dim ablation runs and produce comparison plots.

Reads per-run metrics.json files from the ablation output directory structure:
  <base_dir>/reduced_dim_<D>_seed_<S>/metrics.json

Produces:
  1. A summary CSV with all metrics per run.
  2. Three grouped box+bar plots: NudeNet Total, FID_COCO, CLIP_COCO.

Usage:
  python scripts/plot_reduced_dim_ablation.py \
      --base-dir /shared/results/common/miksa/intact/SD/ablation \
      --output-dir ./ablation_plots
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot reduced_dim ablation results",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Root directory containing reduced_dim_<D>_seed_<S> subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ablation_plots",
        help="Directory to save output plots and CSV.",
    )
    return parser.parse_args()


def discover_runs(base_dir: str) -> List[Dict[str, Any]]:
    """Scan base_dir for reduced_dim_<D>_seed_<S>/metrics.json and parse."""
    pattern = re.compile(r"reduced_dim_(\d+)_seed_(\d+)")
    rows: List[Dict[str, Any]] = []

    base = Path(base_dir)
    if not base.exists():
        print(f"[!] Base directory does not exist: {base}")
        return rows

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        dim = int(m.group(1))
        seed = int(m.group(2))
        metrics_path = entry / "metrics.json"
        if not metrics_path.exists():
            print(f"[W] No metrics.json in {entry}")
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        rows.append({
            "reduced_dim": dim,
            "seed": seed,
            "nudenet_total": data.get("nudenet/Total"),
            "fid_coco": data.get("FID_COCO"),
            "clip_coco": data.get("CLIP_COCO"),
            "ua": data.get("UA"),
        })

    return rows


def save_csv(rows: List[Dict[str, Any]], output_dir: str) -> str:
    """Save aggregated metrics to CSV."""
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Saved summary CSV to {out_path}")
    return out_path


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    higher_is_better: bool,
    output_dir: str,
) -> None:
    """Produce a combined boxplot + mean±SE bar chart for one metric."""
    dims = sorted(df["reduced_dim"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(dims)))

    fig, (ax_box, ax_bar) = plt.subplots(
        2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 2]}, sharex=True
    )
    fig.suptitle(f"Reduced Dim Ablation – {metric_label}", fontsize=14, fontweight="bold")

    # --- Boxplot (per-seed distribution) ---
    grouped = [df.loc[df["reduced_dim"] == d, metric_col].dropna().values for d in dims]
    bp = ax_box.boxplot(
        grouped, positions=list(dims), widths=0.6, patch_artist=True, showfliers=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, d in enumerate(dims):
        runs = df[df["reduced_dim"] == d][metric_col].dropna()
        if len(runs) > 0:
            ax_box.plot(i + 1, runs.mean(), "rD", markersize=6, label="mean" if i == 0 else "")
    ax_box.set_ylabel(metric_label)
    ax_box.set_xticks(list(range(len(dims))))
    ax_box.set_xticklabels([str(d) for d in dims])
    ax_box.legend(loc="upper left")
    ax_box.grid(axis="y", alpha=0.3)

    # --- Bar chart (mean ± SE) ---
    means = []
    sems = []
    for d in dims:
        vals = df[df["reduced_dim"] == d][metric_col].dropna()
        if len(vals) > 0:
            means.append(vals.mean())
            sems.append(vals.sem())
        else:
            means.append(np.nan)
            sems.append(0)

    ax_bar.bar(
        dims, means, yerr=sems, capsize=5, color=colors, edgecolor="black", alpha=0.85,
    )
    if higher_is_better:
        bar_dir = "increase"
    else:
        bar_dir = "decrease"
    ax_bar.set_ylabel(f"Mean ± SE ({bar_dir} = better)")
    ax_bar.set_xlabel("Reduced Dimensionality")
    ax_bar.set_xticks(list(dims))
    ax_bar.set_xticklabels([str(d) for d in dims])
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_file = os.path.join(
        output_dir, f"ablation_{metric_col.replace('/', '_')}.png"
    )
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved plot: {out_file}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = discover_runs(args.base_dir)
    if not rows:
        print("[!] No runs found. Check --base-dir path.")
        return

    print(f"[+] Found {len(rows)} runs.")
    csv_path = save_csv(rows, args.output_dir)

    df = pd.DataFrame(rows)

    plot_metric(df, "nudenet_total", "NudeNet Total (lower = better)", False, args.output_dir)
    plot_metric(df, "fid_coco", "FID COCO (lower = better)", False, args.output_dir)
    plot_metric(df, "clip_coco", "CLIP Score COCO (higher = better)", True, args.output_dir)


if __name__ == "__main__":
    main()
