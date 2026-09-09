#!/usr/bin/env python3
"""
Shared CSV / summarise machinery for the mechanism + design-choice ablation
runners (DDPM and ResNet-18).  Keeps one homogeneous schema
(results/ablations.csv) and renders one booktabs table per experiment into
<results_dir>/tables/*.tex, in the style of the paper's ablation tables.

Experiments covered (see results/README.md):
    exp3  uniform-margin control      exp6  sign-convention sensitivity
    exp4  centre vs width              exp7  percentile alpha
    exp5  protected-region family      exp8  delta_b in the interval legs
    exp9  L_res isolation

Normalisation: every variant's interval part is divided by its own number of
squared terms (env_box: 2, two_corner/two_random: 4, slabs_2k: 4k,
width_only / centre_only: 2 each) so all variants sit on a comparable scale.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import torch

log = logging.getLogger(__name__)

BACKBONES = ("ddpm", "resnet18")


def torch_load(path, map_location="cpu", **kwargs):
    """torch.load with a fallback for torch < 1.13 (no weights_only kwarg)."""
    try:
        return torch.load(path, map_location=map_location,
                          weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)

# One row per (experiment, variant, lambda, seed, backbone).
ABLATIONS_CSV_COLS = [
    "experiment", "backbone", "setting",
    "region_mode", "interval_mode", "alpha", "sign_flip_frac", "include_db",
    "uniform_margin", "include_mean", "include_res",
    "lambda", "fixed_lambda", "seed", "terms", "analytic_matvecs",
    "ua", "ra", "ta", "fid", "fid_backend", "mia",
    "prot_loss_ms", "prot_loss_ms_std",
    "setup_wall_s", "train_wall_s", "total_wall_s", "peak_mem_mb",
    "n_iters", "k", "tparams", "run_dir",
]

# Legacy region-grid schema (regenerated offline by aggregate_rows).
REGIONS_CSV_COLS = [
    "region_mode", "lambda", "seed", "terms", "analytic_matvecs",
    "ua", "ta", "fid", "fid_backend",
    "prot_loss_ms", "prot_loss_ms_std",
    "setup_wall_s", "train_wall_s", "total_wall_s", "peak_mem_mb",
    "n_iters", "k", "tparams", "diag_dirs", "run_dir",
]

# Per-layer diagnostics schema (regenerated offline by aggregate_rows).
DIAGNOSTICS_CSV_COLS = [
    "region_mode", "lambda", "seed", "layer_name", "k",
    "c_lo_norm", "c_hi_norm", "env_asym_ratio",
    "frac_remain_in_A", "frac_remain_in_B", "frac_remain_in_C",
    "frac_remain_in_D", "frac_remain_in_E", "frac_remain_in_F",
    "aniso_A", "aniso_B", "aniso_C", "aniso_D", "aniso_E", "aniso_F",
    "vstar_drift_protected_A", "vstar_drift_envelope_A",
    "vstar_drift_protected_B", "vstar_drift_envelope_B",
    "vstar_drift_protected_C", "vstar_drift_envelope_C",
    "vstar_drift_protected_D", "vstar_drift_envelope_D",
    "vstar_drift_protected_E", "vstar_drift_envelope_E",
    "vstar_drift_protected_F", "vstar_drift_envelope_F",
    "diag_dirs",
]


# Fixed (paper) lambda per backbone/setting, used to label the "fixed" row.
FIXED_LAMBDA = {
    ("ddpm", "classwise"): 5.0,
    ("ddpm", "random"): 5.0,
    ("resnet18", "classwise"): 10.0,
    ("resnet18", "random"): 1.0,
}


# ============================================================================
# Parallel-safe row transport: per-run sidecar files, aggregated offline.
# The runners NEVER append to shared CSVs directly (hundreds of concurrent
# appends would interleave rows); each run writes <run_dir>/rows.json
# {"run": {...}, "diagnostics": [...]} and --summarize aggregates.
# ============================================================================

def run_key(row):
    """Canonical identity of a run: everything that changes loss or metrics."""
    keys = ("experiment", "backbone", "setting", "region_mode", "interval_mode",
            "alpha", "sign_flip_frac", "include_db", "uniform_margin",
            "include_mean", "include_res", "lambda", "seed")
    return tuple(str(row.get(k, "")) for k in keys)


def run_key_suppl(r):
    return "|".join(str(r.get(k, "")) for k in
                    ("experiment", "backbone", "setting", "region_mode",
                     "interval_mode", "alpha", "sign_flip_frac", "include_db",
                     "uniform_margin", "include_mean", "include_res",
                     "lambda", "seed"))


def safe_run_suffix(row_or_ident):
    """Collision-proof run-dir suffix encoding ALL discriminating flags (two
    distinct configurations NEVER share a directory, regardless of clock
    resolution)."""
    def g(k, default=""):
        v = row_or_ident.get(k, default)
        return "" if v is None else str(v)

    def _flag(k, default=0):
        v = row_or_ident.get(k, default)
        if v is None or v == "":
            return 0
        if isinstance(v, bool):
            return int(v)
        s = str(v).strip().lower()
        if s in ("true", "yes", "on", "1"):
            return 1
        if s in ("false", "no", "off", "0"):
            return 0
        return int(float(s))

    return (
        f"{g('experiment')}_{g('region_mode')}_{g('interval_mode', 'full')}"
        f"_a{g('alpha', '5')}_f{g('sign_flip_frac', '0.0')}"
        f"_db{_flag('include_db')}"
        f"_um{_flag('uniform_margin')}"
        f"_mn{_flag('include_mean', 1)}_rs{_flag('include_res', 1)}"
        f"_lam{g('lambda')}_s{g('seed')}"
    )


def write_sidecar(run_dir, run_row, diagnostics=None):
    """Write <run_dir>/rows.json with one unified run row + optional per-layer
    diagnostics rows.  Adds run_dir and a timestamp (aggregation keeps the
    newest row per run_key on resubmission)."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "run": {c: run_row.get(c, "") for c in ABLATIONS_CSV_COLS},
        "diagnostics": diagnostics or [],
    }
    payload["run"]["run_dir"] = run_dir
    payload["run"]["ts"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(run_dir, "rows.json"), "w") as f:
        json.dump(payload, f, indent=1)


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, path)  # atomic: no torn reads by concurrent summarise


def _newest(rows, keyfn):
    by = {}
    for r in rows:
        k = keyfn(r)
        if k not in by or (r.get("ts") or "") > (by[k].get("ts") or ""):
            by[k] = r
    return list(by.values())


def aggregate_rows(results_dir):
    """Scan <results_dir>/runs/*/rows.json and (re)generate regions.csv,
    ablations.csv and diagnostics.csv.  Deterministic, idempotent, and safe to
    run while jobs are still writing (sidecars are per-run files)."""
    runs_root = os.path.join(results_dir, "runs")
    if not os.path.isdir(runs_root):
        log.info(f"no {runs_root}; nothing to aggregate")
        return
    ab, diag = [], []
    n_sidecars = 0
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        p = os.path.join(d, "rows.json")
        if not (os.path.isdir(d) and os.path.exists(p)):
            continue
        try:
            with open(p) as f:
                payload = json.load(f)
        except Exception as exc:
            log.warning(f"skipping unreadable sidecar {p}: {exc}")
            continue
        r = payload.get("run")
        if r:
            ab.append(r)
            n_sidecars += 1
        diag.extend(payload.get("diagnostics", []) or [])

    ab = _newest(ab, run_key)
    reg = [{c: r.get(c, "") for c in REGIONS_CSV_COLS} for r in ab]
    diag = _newest(diag, lambda r: run_key_suppl(r) + ("|" + str(r.get("layer_name", "")) if r.get("layer_name") else ""))
    _write_csv(os.path.join(results_dir, "ablations.csv"), ABLATIONS_CSV_COLS, ab)
    _write_csv(os.path.join(results_dir, "regions.csv"), REGIONS_CSV_COLS, reg)
    _write_csv(os.path.join(results_dir, "diagnostics.csv"), DIAGNOSTICS_CSV_COLS, diag)
    log.info(f"aggregated {n_sidecars} sidecars -> {len(ab)} unique runs, "
             f"{len(diag)} diagnostics rows")


# ============================================================================
# Summarise -> results/tables/<experiment>.tex
# ============================================================================

def _load_ablations(results_dir):
    path = os.path.join(results_dir, "ablations.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dedup(rows, keys=("experiment", "backbone", "setting", "region_mode",
                       "interval_mode", "alpha", "sign_flip_frac", "include_db",
                       "uniform_margin", "include_mean", "include_res",
                       "lambda", "seed")):
    latest = {}
    for r in rows:
        k = tuple(str(r.get(kk, "")) for kk in keys)
        latest[k] = r
    return list(latest.values())


def _load_legacy_regions(results_dir):
    """Legacy region-grid rows (regions.csv, pre-experiment schema) re-tagged
    as exp5 so the region-family table shows all modes."""
    path = os.path.join(results_dir, "regions.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("region_mode"):
                continue
            rows.append({
                "experiment": "exp5",
                "backbone": "ddpm",
                "setting": "classwise",
                "region_mode": r["region_mode"],
                "interval_mode": "full",
                "alpha": 5,
                "sign_flip_frac": 0.0,
                "include_db": 0,
                "uniform_margin": 0,
                "include_mean": 1,
                "include_res": 1,
                "lambda": r.get("lambda", ""),
                "fixed_lambda": int(float(r.get("lambda", 0)) == 5.0),
                "seed": r.get("seed", ""),
                "terms": r.get("terms", ""),
                "analytic_matvecs": r.get("analytic_matvecs", ""),
                "ua": r.get("ua", ""),
                "ra": "",
                "ta": r.get("ta", ""),
                "fid": r.get("fid", ""),
                "fid_backend": r.get("fid_backend", ""),
                "mia": "",
                "prot_loss_ms": r.get("prot_loss_ms", ""),
                "prot_loss_ms_std": r.get("prot_loss_ms_std", ""),
                "setup_wall_s": r.get("setup_wall_s", ""),
                "train_wall_s": r.get("train_wall_s", ""),
                "total_wall_s": r.get("total_wall_s", ""),
                "peak_mem_mb": r.get("peak_mem_mb", ""),
                "n_iters": r.get("n_iters", ""),
                "k": r.get("k", ""),
                "tparams": r.get("tparams", ""),
                "run_dir": r.get("run_dir", ""),
            })
    return rows


def _merge_exp5(exp5_rows, legacy_rows):
    """Keep ablations.csv rows (newest regime) and fill region modes from the
    legacy grid; dedup by (region_mode, lambda, seed)."""
    merged = {}
    for r in exp5_rows:
        key = (r.get("region_mode"), str(r.get("lambda")), str(r.get("seed")))
        merged[key] = r
    for r in legacy_rows:
        key = (r["region_mode"], str(r.get("lambda")), str(r.get("seed")))
        if key not in merged:
            merged[key] = r
    return list(merged.values())


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.{nd}f}"


def _mean_std(vals):
    nums = []
    for v in vals:
        if v is None or v == "":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            nums.append(f)
    if not nums:
        return float("nan"), float("nan")
    a = np.asarray(nums, dtype=float)
    return float(a.mean()), float(a.std())


# Display labels used in table headers -> column names in ablations.csv.
METRIC_KEYS = {"UA": "ua", "RA": "ra", "TA": "ta", "FID": "fid",
               "TParams": "tparams", "terms": "terms"}


def _agg_tex(entries, cols):
    """Mean +- std cells over seeds for the requested metric columns."""
    out = []
    for c in cols:
        k = METRIC_KEYS.get(c, c)
        m, s = _mean_std([e.get(k) for e in entries])
        out.append(rf"{_fmt(m)}$\pm${_fmt(s)}")
    return " & ".join(out)


def _experiment_table(experiment, rows, backbone, metrics,
                      variant_col, variant_label,
                      lambda_label="$\lambda_\mathrm{int}$",
                      note="", extra_cols=(), label_map=None):
    """Standard ablation table: variant rows at fixed lambda + best-lambda-per
    variant.  `metrics` is the list of metric columns to tabulate."""
    fixed_lam = FIXED_LAMBDA.get((backbone, "classwise"))
    rows_by_v = {}
    for r in rows:
        v = str(r.get(variant_col, ""))
        if label_map:
            v = label_map.get(v, v)
        rows_by_v.setdefault(v, []).append(r)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    fixed_txt = _fmt(fixed_lam, 1).rstrip("0").rstrip(".")
    lines.append(rf"\caption{{{experiment} on {backbone} ({variant_label}). "
                 rf"Mean $\pm$ std over 3 seeds. \texttt{{fixed}} = "
                 rf"$\lambda_\mathrm{{int}}={fixed_txt}$; \texttt{{best}} = "
                 rf"best $\lambda_\mathrm{{int}}$ per variant. "
                 rf"{note}\label{{tab:{experiment}-{backbone}}}}}")
    lines.append(r"\small")
    n_metric = len(metrics)
    align = "l c " + "c " * (n_metric + len(extra_cols))
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    header = (rf"Variant & {lambda_label} & {' & '.join(metrics)}"
              + (" & " + " & ".join(extra_cols) if extra_cols else "") + r" \\")
    lines.append(header)
    lines.append(r"\midrule")

    best_per_variant = {}
    for v, entries in rows_by_v.items():
        fixed = [e for e in entries
                 if str(e.get("fixed_lambda", "")).lower() in ("1", "true", "yes")]
        fixed = fixed or [e for e in entries if float(str(e.get("lambda", "nan"))) == fixed_lam]
        if fixed:
            cells = _agg_tex(fixed, metrics)
            if extra_cols:
                cells += " & " + _agg_tex(fixed, list(extra_cols))
            lines.append(rf"{v} (fixed) & "
                         rf"{_fmt(fixed_lam, 1).rstrip('0').rstrip('.')} & "
                         rf"{cells} \\")

        # best-per-variant = the lambda whose CELL MEAN (over seeds) maximises
        # the composite — not the best single seed (which can lie below the
        # mean and would disagree with the table's mean +/- std cells).
        by_lam = {}
        for e in entries:
            by_lam.setdefault(float(str(e.get("lambda", "nan"))), []).append(e)

        def _cell_score(es):
            u, _ = _mean_std([float(x.get("ua", "nan")) for x in es])
            t, _ = _mean_std([float(x.get("ta", "nan")) for x in es])
            if u != u or t != t:
                return float("-inf")
            f, _ = _mean_std([float(x.get("fid", "nan")) for x in es])
            concrete = f if f == f else 0.0
            return 1.5 * u + t - concrete / 3.0

        best_group = max(by_lam.values(), key=_cell_score)
        if best_group:
            best_per_variant[v] = best_group
            lam = float(str(best_group[0].get("lambda", "nan")))
            cells = _agg_tex(best_group, metrics)
            if extra_cols:
                cells += " & " + _agg_tex(best_group, list(extra_cols))
            lines.append(rf"{v} (best) & {_fmt(lam, 1).rstrip('0').rstrip('.')} & "
                         rf"{cells} \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return lines


def summarize_ablations(results_dir):
    """Render one tex table per present experiment into results/tables/ and
    print the console summary."""
    rows = _load_ablations(results_dir)
    if not rows:
        log.info(f"No ablations.csv at {results_dir}; nothing to summarise")
        return

    rows = _dedup(rows)
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    legacy_rows = _load_legacy_regions(results_dir)
    has_legacy = bool(legacy_rows)
    if has_legacy:
        log.info(f"found {len(legacy_rows)} legacy region-grid rows")

    experiments = sorted({r["experiment"] for r in rows if r.get("experiment")})
    if "exp5" not in experiments and has_legacy:
        experiments = sorted(experiments + ["exp5"])
    for experiment in experiments:
        erows = [r for r in rows if r["experiment"] == experiment]
        backbones = sorted({r["backbone"] for r in erows})
        for backbone in backbones:
            brows = [r for r in erows if r["backbone"] == backbone]
            if experiment == "exp3":
                lines = _experiment_table(
                    "Experiment 3: uniform-margin control (is the data-adaptive "
                    "weighting doing the work?)",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="uniform_margin",
                    variant_label="uniform margins vs standard bounds",
                    extra_cols=("TParams",) if backbone == "resnet18" else ("terms",),
                    label_map={"0": "standard", "1": "uniform"},
                )
            elif experiment == "exp4":
                lines = _experiment_table(
                    "Experiment 4: centre vs width (which half of Lemma 1 earns "
                    "its place)",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="interval_mode",
                    variant_label="interval-mode variants",
                )
            elif experiment == "exp5":
                if backbone == "ddpm":
                    real = [r for r in brows if r.get("region_mode")]
                    brows = _merge_exp5(real, legacy_rows) if has_legacy else real
                lines = _experiment_table(
                    "Experiment 5: protected-region construction",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="region_mode",
                    variant_label="region modes",
                )
            elif experiment == "exp6":
                lines = _experiment_table(
                    "Experiment 6: sign-convention sensitivity "
                    "(fraction of U\\_forget rows flipped)",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="sign_flip_frac",
                    variant_label="sign-flip fractions",
                )
            elif experiment == "exp7":
                lines = _experiment_table(
                    "Experiment 7: percentile alpha (control-region size vs "
                    "margin width)",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="alpha",
                    variant_label="percentile alpha",
                )
            elif experiment == "exp8":
                lines = _experiment_table(
                    "Experiment 8: delta\\_b inside the interval corner legs "
                    "(InTAct Eqs. 35-36)",
                    brows, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="include_db",
                    variant_label="include delta\\_b",
                    label_map={"0": "off", "1": "on"},
                )
            elif experiment == "exp9":
                brows9 = []
                for r in brows:
                    combo = []
                    if str(r.get("interval_mode", "")) != "off":
                        combo.append("intervals")
                    if str(r.get("include_mean", "1")).lower() in ("1", "true"):
                        combo.append("Lmean")
                    if str(r.get("include_res", "1")).lower() in ("1", "true"):
                        combo.append("Lres")
                    r = dict(r)
                    r["_combo"] = "+".join(combo) or "none"
                    brows9.append(r)
                lines = _experiment_table(
                    "Experiment 9: L\\_res isolation (component attribution)",
                    brows9, backbone,
                    metrics=["UA", "TA", "FID" if backbone == "ddpm" else "RA"],
                    variant_col="_combo",
                    variant_label="SVD components",
                )
            else:
                continue
            out = os.path.join(tables_dir, f"{experiment}_{backbone}.tex")
            with open(out, "w") as f:
                f.write("\n".join(lines) + "\n")
            log.info(f"Wrote {out}")
            print(f"\n[{experiment} | {backbone}] "
                  f"{len(brows)} unique runs, table {out}")
            print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()
    summarize_ablations(args.results_dir)