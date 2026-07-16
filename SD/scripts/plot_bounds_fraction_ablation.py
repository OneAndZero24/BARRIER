#!/usr/bin/env python3
"""Aggregate metrics from bounds-fraction ablation runs and produce comparison plots.

Two ablations are supported:
  forget_fraction  – vary % of forget set used for bounds (remain at 100%)
  remain_fraction  – vary % of remain set used for bounds (forget at 100%)

Directory layout:
  <base_dir>/forget_frac_<PCT>_seed_<S>/metrics.json
  <base_dir>/remain_frac_<PCT>_seed_<S>/metrics.json

Produces:
  1. Summary CSV per ablation.
  2. Three grouped box+bar plots per ablation: NudeNet Total, FID_COCO, CLIP_COCO.
  3. Side-by-side comparison plots (forget vs remain fraction).

Usage:
  python scripts/plot_bounds_fraction_ablation.py \
      --base-dir /shared/results/common/miksa/intact/SD/ablation \
      --output-dir ./ablation_plots
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTION_LABELS: Dict[int, str] = {10: "10%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot bounds-fraction ablation results")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Root directory containing forget_frac_<PCT>_seed_<S> and remain_frac_<PCT>_seed_<S> subfolders.")
    parser.add_argument("--output-dir", type=str, default="./ablation_plots",
                        help="Directory to save output plots and CSV.")
    return parser.parse_args()


def discover_runs(base_dir: str, prefix: str) -> List[Dict[str, Any]]:
    pattern = re.compile(rf"{prefix}_(\d+)_seed_(\d+)")
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
        pct = int(m.group(1))
        seed = int(m.group(2))
        metrics_path = entry / "metrics.json"
        if not metrics_path.exists():
            print(f"[W] No metrics.json in {entry}")
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        rows.append({
            "fraction_pct": pct,
            "seed": seed,
            "nudenet_total": data.get("nudenet/Total"),
            "fid_coco": data.get("FID_COCO"),
            "clip_coco": data.get("CLIP_COCO"),
            "ua": data.get("UA"),
        })
    return rows


def save_csv(df: pd.DataFrame, output_dir: str, name: str) -> str:
    out_path = os.path.join(output_dir, f"ablation_{name}_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Saved summary CSV to {out_path}")
    return out_path


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    higher_is_better: bool,
    output_dir: str,
    tag: str,
    xlabel: str = "Bounds Fraction",
) -> None:
    pcts = sorted(df["fraction_pct"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(pcts)))
    x_labels = [FRACTION_LABELS.get(p, f"{p}%") for p in pcts]

    fig, (ax_box, ax_bar) = plt.subplots(
        2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2]}, sharex=True
    )
    fig.suptitle(f"{tag} – {metric_label}", fontsize=14, fontweight="bold")

    grouped = [df.loc[df["fraction_pct"] == p, metric_col].dropna().values for p in pcts]
    bp = ax_box.boxplot(
        grouped, positions=list(range(1, len(pcts) + 1)), widths=0.6,
        patch_artist=True, showfliers=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i in range(len(pcts)):
        runs = grouped[i]
        if len(runs) > 0:
            ax_box.plot(i + 1, runs.mean(), "rD", markersize=6,
                        label="mean" if i == 0 else "")
    ax_box.set_ylabel(metric_label)
    ax_box.set_xticks(list(range(1, len(pcts) + 1)))
    ax_box.set_xticklabels(x_labels)
    ax_box.legend(loc="upper left")
    ax_box.grid(axis="y", alpha=0.3)

    means = [g.mean() if len(g) > 0 else np.nan for g in grouped]
    sems = [g.sem() if len(g) > 0 else 0.0 for g in grouped]
    ax_bar.bar(
        x_labels, means, yerr=sems, capsize=5, color=colors, edgecolor="black", alpha=0.85,
    )
    ax_bar.set_ylabel("Mean ± SE (" + ("increase" if higher_is_better else "decrease") + " = better)")
    ax_bar.set_xlabel(xlabel)
    ax_bar.set_xticklabels(x_labels)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    safe_name = metric_col.replace("/", "_")
    out_file = os.path.join(output_dir, f"ablation_{tag}_{safe_name}.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved plot: {out_file}")


def plot_comparison(
    df_forget: pd.DataFrame,
    df_remain: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    higher_is_better: bool,
    output_dir: str,
) -> None:
    pcts = sorted(df_forget["fraction_pct"].unique())
    x_labels = [FRACTION_LABELS.get(p, f"{p}%") for p in pcts]
    x = np.arange(len(pcts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Forget vs Remain Bounds Fraction – {metric_label}", fontsize=13, fontweight="bold")

    for df, label, offset, color in [
        (df_forget, "Forget fraction", -width / 2, "#E26B6B"),
        (df_remain, "Remain fraction", +width / 2, "#6B9DE2"),
    ]:
        means, sems = [], []
        for p in pcts:
            vals = df[df["fraction_pct"] == p][metric_col].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(vals.sem() if len(vals) > 0 else 0.0)
        ax.bar(x + offset, means, width, yerr=sems, capsize=4,
               color=color, edgecolor="black", alpha=0.85, label=label)

    ax.set_ylabel(metric_label)
    ax.set_xlabel("Bounds Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    safe_name = metric_col.replace("/", "_")
    out = os.path.join(output_dir, f"ablation_comparison_{safe_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved comparison plot: {out}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_forget = pd.DataFrame(discover_runs(args.base_dir, "forget_frac"))
    df_remain = pd.DataFrame(discover_runs(args.base_dir, "remain_frac"))

    if df_forget.empty and df_remain.empty:
        print("[!] No runs found. Check --base-dir path.")
        return

    for df, tag in [(df_forget, "forget_fraction"), (df_remain, "remain_fraction")]:
        if df.empty:
            print(f"[!] No runs found for {tag}, skipping.")
            continue
        print(f"[+] Found {len(df)} runs for {tag}.")
        save_csv(df, args.output_dir, tag)
        plot_metric(df, "nudenet_total", "NudeNet Total (lower = better)", False, args.output_dir, tag)
        plot_metric(df, "fid_coco", "FID COCO (lower = better)", False, args.output_dir, tag)
        plot_metric(df, "clip_coco", "CLIP Score COCO (higher = better)", True, args.output_dir, tag)

    if not df_forget.empty and not df_remain.empty:
        print(f"[+] Generating comparison plots (forget vs remain).")
        plot_comparison(df_forget, df_remain, "nudenet_total", "NudeNet Total (lower = better)", False, args.output_dir)
        plot_comparison(df_forget, df_remain, "fid_coco", "FID COCO (lower = better)", False, args.output_dir)
        plot_comparison(df_forget, df_remain, "clip_coco", "CLIP Score COCO (higher = better)", True, args.output_dir)


if __name__ == "__main__":
    main()