#!/usr/bin/env python3
"""
Collect DDPM InTAct lambda_interval-sweep results and render the three
pairwise trade-off curves (UA vs RA, RA vs FID, UA vs FID), one point per
lambda, connected by a line and annotated with the lambda value.

Reads the per-run sidecars written by ablation_regions.py
(<results_dir>/runs/*/rows.json), aggregates mean +- std over seeds, and
writes PNG figures plus a summary CSV into <out_dir> (default
<results_dir>/plots).

Note on metrics: for the DDPM class-forgetting setting the retain accuracy is
stored in the "ta" column (the "ra" column is reserved for ResNet-18
classification); RA below refers to that retain/remaining-class accuracy.

Usage:
    python scripts/plot_lambda_sweep.py \
        --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
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

# Swept values must match the SLURM array (sorted ascending for the curve).
LAMBDAS = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0]
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


def _fmt(v):
    return f"{v:.3f}" if v == v else "-"


def _lam_label(lam):
    s = f"{lam:g}"
    return s


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


def _curve(agg, ykey, xkey):
    """Return ordered (x_mean, x_err, y_mean, y_err, labels) for non-NaN pts."""
    xs, xe, ys, ye, lbls = [], [], [], [], []
    for lam in LAMBDAS:
        if lam not in agg:
            continue
        a = agg[lam]
        xm = a[xkey][0]
        ym = a[ykey][0]
        if xm != xm or ym != ym:
            continue
        xs.append(xm)
        xe.append(a[xkey][1] if a[xkey][1] == a[xkey][1] else 0.0)
        ys.append(ym)
        ye.append(a[ykey][1] if a[ykey][1] == a[ykey][1] else 0.0)
        lbls.append(lam)
    return np.asarray(xs), np.asarray(xe), np.asarray(ys), np.asarray(ye), lbls


def plot_curve(agg, ykey, xkey, xlabel, ylabel, title, out_path):
    xs, xe, ys, ye, lbls = _curve(agg, ykey, xkey)
    if len(xs) == 0:
        print(f"[skip] no data for {title}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="-o", color="#1f77b4",
                lw=1.5, ms=5, capsize=3, zorder=2)
    for x, y, lam in zip(xs, ys, lbls):
        ax.annotate(rf"$\lambda$={_lam_label(lam)}", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default=None,
                   help="output dir for figures/CSV (default <results_dir>/plots)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")

    rows = load_runs(args.results_dir)
    print(f"found {len(rows)} runs for experiment={EXPERIMENT}")
    if not rows:
        print("nothing to plot; run the SLURM array first")
        return

    agg = aggregate(rows)
    present = sorted(agg)
    print(f"lambdas present: {[f'{l:g}' for l in present]}")

    summary_path = write_summary(agg, out_dir)
    print(f"wrote {summary_path}")

    plot_curve(agg, "ua", "ra", "RA (retain accuracy)",
               "UA (unlearning accuracy)",
               "UA vs RA (lambda_interval sweep)", os.path.join(out_dir, "ua_vs_ra.png"))
    plot_curve(agg, "ra", "fid", "FID",
               "RA (retain accuracy)",
               "RA vs FID (lambda_interval sweep)", os.path.join(out_dir, "ra_vs_fid.png"))
    plot_curve(agg, "ua", "fid", "FID",
               "UA (unlearning accuracy)",
               "UA vs FID (lambda_interval sweep)", os.path.join(out_dir, "ua_vs_fid.png"))


if __name__ == "__main__":
    main()
