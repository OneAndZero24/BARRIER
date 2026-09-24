#!/usr/bin/env python3
"""
Collect DDPM InTAct lambda_interval-sweep results and render three bar plots
(UA, RA, FID) — one bar per lambda, mean over seeds with +- std error bars.

Reads the per-run sidecars written by ablation_regions.py
(<results_dir>/runs/*/rows.json), aggregates mean +- std over seeds, and
writes PNG figures plus a summary CSV into <out_dir> (default
<results_dir>/plots).  Alternatively reads an existing summary.csv via
--from_csv.

Note on metrics: for the DDPM class-forgetting setting the retain accuracy is
stored in the "ta" column (the "ra" column is reserved for ResNet-18
classification); RA below refers to that retain/remaining-class accuracy.

Usage:
    python scripts/plot_lambda_sweep.py \
        --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
    python scripts/plot_lambda_sweep.py \
        --from_csv ~/Downloads/summary.csv --exclude 2 5
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

# Canonical sweep set (controls inclusion in plots + summary-CSV row order).
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 30.0, 50.0, 100.0]
EXPERIMENT = "lambda_sweep"


def load_runs(results_dir):
    """Return the list of 'run' dicts for this experiment from sidecars."""
    runs_root = Path(results_dir) / "runs"
    rows = []
    if not runs_root.is_dir():
        return rows
    for d in sorted(runs_root.iterdir()):
        p = d / "rows.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        r = payload.get("run")
        if r and r.get("experiment") == EXPERIMENT:
            rows.append(r)
    return rows


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if x == x else float("nan")


def _mean_std(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def aggregate(rows):
    """Group runs by lambda -> {ua, ra, fid} mean/std + seed count."""
    by_lam = {}
    for r in rows:
        lam = _f(r.get("lambda"))
        if lam != lam:
            continue
        by_lam.setdefault(lam, []).append({
            "ua": _f(r.get("ua")),
            "ra": _f(r.get("ta")),  # DDPM retain accuracy lives in "ta"
            "fid": _f(r.get("fid")),
        })

    out = {}
    for lam, entries in by_lam.items():
        out[lam] = {
            "n_seeds": len(entries),
            "ua": _mean_std([e["ua"] for e in entries]),
            "ra": _mean_std([e["ra"] for e in entries]),
            "fid": _mean_std([e["fid"] for e in entries]),
        }
    return out


def load_summary_csv(csv_path):
    """Load the aggregated {lam: {...}} structure from a summary.csv written
    by write_summary (same schema as aggregate())."""
    agg = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lam = float(row["lambda"])
            agg[lam] = {
                "n_seeds": int(float(row["n_seeds"])),
                "ua": (float(row["UA_mean"]), float(row["UA_std"])),
                "ra": (float(row["RA_mean"]), float(row["RA_std"])),
                "fid": (float(row["FID_mean"]), float(row["FID_std"])),
            }
    return agg


def _fmt(v):
    return f"{v:.3f}" if v == v else "-"


def _lam_label(lam):
    return f"{lam:g}"


def write_summary(agg, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "n_seeds",
                    "UA_mean", "UA_std",
                    "RA_mean", "RA_std",
                    "FID_mean", "FID_std"])
        for lam in LAMBDAS:
            if lam not in agg:
                continue
            a = agg[lam]
            w.writerow([
                lam, a["n_seeds"],
                _fmt(a["ua"][0]), _fmt(a["ua"][1]),
                _fmt(a["ra"][0]), _fmt(a["ra"][1]),
                _fmt(a["fid"][0]), _fmt(a["fid"][1]),
            ])
    return path


def bar_plot(agg, key, ylabel, title, out_path, color="#1f77b4"):
    """Bar chart of one metric per lambda, mean +- std over seeds."""
    lams = [lam for lam in LAMBDAS if lam in agg]
    if not lams:
        print(f"[skip] no data for {title}")
        return

    vals = [agg[lam][key][0] for lam in lams]
    errs = [agg[lam][key][1] if agg[lam][key][1] == agg[lam][key][1] else 0.0
            for lam in lams]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(lams))
    ax.bar(x, vals, yerr=errs, width=0.55, color=color,
           edgecolor="black", linewidth=0.6, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_lam_label(lam) for lam in lams])
    ax.set_xlabel(r"$\lambda_\mathrm{interval}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    top = max(v + e for v, e in zip(vals, errs))
    bot = min([v - e for v, e in zip(vals, errs)] + [0.0])
    pad = 0.03 * max(top - bot, 1e-9) + 0.01
    ax.set_ylim(bot - pad, top + pad * 4)
    for xi, v, e in zip(x, vals, errs):
        ax.text(float(xi), v + e + pad, f"{v:.3f}", ha="center", fontsize=8)

    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default=None,
                   help="path to aggregated sidecars directory "
                        "(optional if --from_csv is given)")
    p.add_argument("--from_csv", default=None,
                   help="read lambdas from a summary.csv instead of sidecars "
                        "(e.g. already downloaded summary.csv)")
    p.add_argument("--exclude", nargs="+", type=float, default=[],
                   help="lambda values to drop from the plots")
    p.add_argument("--out_dir", default=None,
                   help="output dir for figures/CSV (default <results_dir>/plots)")
    args = p.parse_args()

    if not args.from_csv and not args.results_dir:
        p.error("either --from_csv or --results_dir is required")

    out_dir = args.out_dir or os.path.join(args.results_dir or ".", "plots")

    if args.from_csv:
        agg = load_summary_csv(args.from_csv)
        print(f"loaded {len(agg)} lambdas from {args.from_csv}")
    else:
        rows = load_runs(args.results_dir)
        print(f"found {len(rows)} runs for experiment={EXPERIMENT}")
        if not rows:
            print("nothing to plot; run the SLURM array first")
            return
        agg = aggregate(rows)

    agg = {lam: agg[lam] for lam in agg if lam in LAMBDAS}
    for lam in args.exclude:
        agg.pop(lam, None)
    present = sorted(agg)
    print(f"lambdas plotted: {[f'{l:g}' for l in present]}")
    if not present:
        print("nothing left to plot after exclusions")
        return

    summary_path = write_summary(agg, out_dir)
    print(f"wrote {summary_path}")

    bar_plot(agg, "ua", "UA (unlearning accuracy)",
             "UA by lambda_interval", os.path.join(out_dir, "ua.png"))
    bar_plot(agg, "ra", "RA (retain accuracy)",
             "RA by lambda_interval", os.path.join(out_dir, "ra.png"))
    bar_plot(agg, "fid", "FID",
             "FID by lambda_interval", os.path.join(out_dir, "fid.png"))


if __name__ == "__main__":
    main()