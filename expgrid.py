#!/usr/bin/env python3
"""
Deterministic grid definitions for the mechanism/design-choice ablation
experiments (Experiments 3-9), shared by the DDPM and ResNet-18 SLURM arrays.

Rows are (experiment, region_mode, interval_mode, alpha, sign_flip_frac,
include_db, uniform_margin, include_mean, include_res, lambda, seed).

  exp3  uniform-margin control        (two_corner x 6L x 3s)
  exp4  centre vs width                (width/centre x 6L x 3s)
  exp5  protected-region family        (env_box x 6L x 3s; the other modes
                                        exist in the earlier region grid)
  exp6  sign-convention sensitivity    (flip frac 0/0.25/0.5 x 3s, lambda fixed)
  exp7  percentile alpha               (alpha 1/5/10 x 6L x 3s)
  exp8  delta_b in the interval legs   (off/on x 6L x 3s)
  exp9  L_res isolation                (3 component combos x 3s, lambda fixed)

Usage:  python expgrid.py <backbone/setting> <index>     (index in 0..N-1)
        python expgrid.py <backbone/setting> --count
"""

import sys

LAMBDA_SWEEP = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
SEEDS = (0, 1, 2)


def _grid(fixed_lambda):
    rows = []
    # exp3: uniform margin vs standard bounds
    for lam in LAMBDA_SWEEP:
        for seed in SEEDS:
            rows.append(("exp3", "two_corner", "full", 5, 0.0, 0, 1, 1, 1, lam, seed))
    # exp4: interval modes
    for im in ("width_only", "centre_only"):
        for lam in LAMBDA_SWEEP:
            for seed in SEEDS:
                rows.append(("exp4", "two_corner", im, 5, 0.0, 0, 0, 1, 1, lam, seed))
    # exp5: env_box (cheap exact-complement variant)
    for lam in LAMBDA_SWEEP:
        for seed in SEEDS:
            rows.append(("exp5", "env_box", "full", 5, 0.0, 0, 0, 1, 1, lam, seed))
    # exp6: sign-flip fractions at the fixed lambda
    for frac in (0.0, 0.25, 0.5):
        for seed in SEEDS:
            rows.append(("exp6", "two_corner", "full", 5, frac, 0, 0, 1, 1,
                         fixed_lambda, seed))
    # exp7: percentile alpha
    for alpha in (1, 5, 10):
        for lam in LAMBDA_SWEEP:
            for seed in SEEDS:
                rows.append(("exp7", "two_corner", "full", alpha, 0.0, 0, 0,
                             1, 1, lam, seed))
    # exp8: delta_b in the interval legs
    for db in (0, 1):
        for lam in LAMBDA_SWEEP:
            for seed in SEEDS:
                rows.append(("exp8", "two_corner", "full", 5, 0.0, db, 0, 1,
                             1, lam, seed))
    # exp9: component isolation at the fixed lambda
    combos = [
        ("off", 0, 1),    # L_res only
        ("full", 0, 1),   # SVD + intervals + L_res, no L_mean
        ("full", 1, 0),   # SVD + intervals + L_mean, no L_res
    ]
    for im, mean, res in combos:
        for seed in SEEDS:
            rows.append(("exp9", "two_corner", im, 5, 0.0, 0, 0, mean, res,
                         fixed_lambda, seed))
    return rows


GRIDS = {
    "ddpm": _grid(5.0),
    "resnet18-classwise": _grid(10.0),
    "resnet18-random": _grid(1.0),
}


def to_flags(row):
    exp, region, interval, alpha, frac, db, uni, mean, res, lam, seed = row
    flags = [f"--experiment {exp}", f"--region_mode {region}",
             f"--interval_mode {interval}", f"--alpha {alpha}",
             f"--sign_flip_frac {frac}"]
    if db:
        flags.append("--include_db")
    if uni:
        flags.append("--uniform_margin")
    if not mean:
        flags.append("--no_include_mean")
    if not res:
        flags.append("--no_include_res")
    flags.append(f"--lambda {lam}")
    flags.append(f"--seed {seed}")
    return " ".join(flags)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    grid = sys.argv[1]
    if grid not in GRIDS:
        print(f"unknown grid {grid!r}; choose one of {sorted(GRIDS)}",
              file=sys.stderr)
        sys.exit(2)
    rows = GRIDS[grid]
    if len(sys.argv) == 3 and sys.argv[2] == "--count":
        print(len(rows))
        return
    idx = int(sys.argv[2])
    if not (0 <= idx < len(rows)):
        print(f"index {idx} out of range 0..{len(rows) - 1}", file=sys.stderr)
        sys.exit(2)
    print(to_flags(rows[idx]))


if __name__ == "__main__":
    main()