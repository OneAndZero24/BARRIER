#!/usr/bin/env python3
"""
Recompute FID on a 500/class subset of the ALREADY-GENERATED fid samples
for the ddpm-fid-repro lambda sweep (35 runs) - no re-training, no
re-sampling.  Reference dataset (cifar10_without_label_0) is 500/class
(4500 images), so a 500/class sample side gives a matched 500-vs-500 FID.

How the 500/class subset is taken: the flat fid_samples_guidance_* dirs
hold contiguous 5000-image blocks per class (class 1 = files 0-4999,
class 2 = 5000-9999, ...), so by default we take the first N files of
each block (--method block).  --method random takes a seeded random
sample of N per block instead.

FID uses the ORIGINAL TF Inception-V3 graph (evaluator.py, pool_3
2048-dim - same code path as compute_fid_full.py); reference activations
are computed once and cached to <results_dir>/.fid500_ref_stats.npz.

Results:
  * each run dir gets rows.json["fid_<N>"] (plus "fid" = original 5000/class
    vs the 500/class reference)
  * `--collect` aggregates all runs into
        <results_dir>/fid<N>_grid.csv   (per-run: lambda, seed, fid_N, fid)
        <results_dir>/fid<N>_plot.csv   (per-lambda: n_seeds, FID_mean, FID_std)

Usage:
    python scripts/recompute_fid_subset.py --results_dir /path/to/lambda_sweep
    python scripts/recompute_fid_subset.py --results_dir /path -n 500 --method random --seed 0
    python scripts/recompute_fid_subset.py --results_dir /path --ref-stats-only  # warm cache
    python scripts/recompute_fid_subset.py --results_dir /path --run-dir <one run>
    python scripts/recompute_fid_subset.py --results_dir /path --collect   # after compute
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_DIR_DEFAULT = "/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"
FID_GLOBS = (
    "fid_samples_guidance_*_excluded_class_*",
    "fid_samples_without_label_*_guidance_*",
    "fid_samples",
)
NUM_CLASSES = 9          # classes 1..9 (label 0 forgotten)
SAMPLE_BLOCK = 5000      # samples saved per class


def find_fid_dir(run_dir):
    for ts in sorted(glob.glob(os.path.join(run_dir, "output", "*"))):
        if not os.path.isdir(ts):
            continue
        for pat in FID_GLOBS:
            hits = sorted(glob.glob(os.path.join(ts, pat)))
            if hits and os.path.isdir(hits[0]) and os.listdir(hits[0]):
                return hits[0]
    return None


def make_subset_dir(sample_dir, n_per_class, method, seed):
    files = sorted(
        (p for p in glob.glob(os.path.join(sample_dir, "*.png"))
         if os.path.isfile(p)),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    expected = NUM_CLASSES * SAMPLE_BLOCK
    if len(files) < expected:
        raise SystemExit(f"{sample_dir}: expected {expected} pngs, found {len(files)}")
    rng = random.Random(seed)
    chosen = []
    for c in range(NUM_CLASSES):
        block = files[c * SAMPLE_BLOCK:(c + 1) * SAMPLE_BLOCK]
        if method == "random":
            chosen.extend(rng.sample(block, n_per_class))
        else:
            chosen.extend(block[:n_per_class])

    # keep the subset dir on the SAME filesystem as the samples (os.link
    # fails across devices, e.g. /shared vs /tmp)
    tmp = os.path.join(os.path.dirname(sample_dir),
                       f".fid500_subset_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    for i, p in enumerate(chosen):
        try:
            os.link(p, os.path.join(tmp, f"{i}.png"))
        except OSError:
            shutil.copy2(p, os.path.join(tmp, f"{i}.png"))
    return tmp


def compute_fid(ref_dir, sample_dir, stats_cache):
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, FIDStatistics, read_images_folder

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if stats_cache and os.path.exists(stats_cache):
        d = dict(np_load(stats_cache))
        ref_stats = FIDStatistics(d["mu"], d["sigma"])
        print(f"  ref stats loaded from {stats_cache}")
    else:
        ref_arr = read_images_folder(ref_dir)
        print(f"  reference: {len(ref_arr)} images")
        sess = tf.Session(config=config)
        evaluator = Evaluator(sess)
        evaluator.warmup()
        ref_acts = evaluator.read_activations(ref_arr)
        ref_stats, _ = evaluator.read_statistics(ref_acts)
        sess.close()
        if stats_cache:
            os.makedirs(os.path.dirname(stats_cache), exist_ok=True)
            np_savez(stats_cache, mu=ref_stats.mu, sigma=ref_stats.sigma)
            print(f"  ref stats cached to {stats_cache}")

    sample_arr = read_images_folder(sample_dir)
    print(f"  sample subset: {len(sample_arr)} images")
    sess = tf.Session(config=config)
    evaluator = Evaluator(sess)
    evaluator.warmup()
    sample_acts = evaluator.read_activations(sample_arr)
    sample_stats, _ = evaluator.read_statistics(sample_acts)
    fid = float(sample_stats.frechet_distance(ref_stats))
    sess.close()
    return fid


def parse_lam_seed(run_dir):
    name = os.path.basename(run_dir)          # lam<lam>_seed<seed>
    lam_s, _, seed_s = name.partition("_seed")
    return float(lam_s[len("lam"):]), int(seed_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT)
    parser.add_argument("-n", "--n-per-class", type=int, default=500)
    parser.add_argument("--method", choices=["block", "random"], default="block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collect", action="store_true",
                        help="aggregate rows.json fid_<N> into CSVs and exit")
    parser.add_argument("--ref-stats-only", action="store_true",
                        help="only compute+cache the reference activations, then exit")
    parser.add_argument("--run-dir", default=None,
                        help="process only this run dir (array mode)")
    args = parser.parse_args()

    if args.collect:
        return collect(args.results_dir)

    if args.n_per_class > SAMPLE_BLOCK:
        raise SystemExit(f"-n {args.n_per_class} > {SAMPLE_BLOCK} available per class")
    if not os.path.isdir(args.ref_dir):
        raise SystemExit(f"ref dataset not found: {args.ref_dir}")

    stats_cache = os.path.join(
        args.results_dir,
        f".fid_ref_stats_{os.path.basename(os.path.normpath(args.ref_dir))}.npz")

    if args.ref_stats_only:
        import tensorflow.compat.v1 as tf
        from evaluator import Evaluator, read_images_folder
        print(f"warmup: caching reference stats for {args.ref_dir} -> {stats_cache}")
        ref_arr = read_images_folder(args.ref_dir)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        evaluator = Evaluator(sess)
        evaluator.warmup()
        ref_acts = evaluator.read_activations(ref_arr)
        ref_stats, _ = evaluator.read_statistics(ref_acts)
        sess.close()
        os.makedirs(os.path.dirname(stats_cache), exist_ok=True)
        np_savez(stats_cache, mu=ref_stats.mu, sigma=ref_stats.sigma)
        print(f"cached reference stats: {stats_cache}")
        return

    run_dirs = sorted(glob.glob(os.path.join(args.results_dir, "runs", "lam*_seed*")))
    if args.run_dir:
        run_dirs = [d for d in run_dirs if os.path.normpath(d) == os.path.normpath(args.run_dir)]
        if not run_dirs:
            raise SystemExit(f"run dir not found: {args.run_dir}")
    if not run_dirs:
        raise SystemExit(f"no runs under {args.results_dir}/runs/")

    for run_dir in run_dirs:
        side = os.path.join(run_dir, "rows.json")
        if not os.path.exists(side):
            print(f"SKIP {run_dir}: no rows.json")
            continue
        row = json.load(open(side))
        fid_dir = find_fid_dir(run_dir)
        if fid_dir is None:
            print(f"SKIP {run_dir}: no fid_samples dir found")
            continue
        lam, seed = parse_lam_seed(run_dir)
        print(f"== run {os.path.basename(run_dir)}  (lam={lam}, seed={seed})")
        sub = make_subset_dir(fid_dir, args.n_per_class, args.method, args.seed)
        try:
            fid = compute_fid(args.ref_dir, sub, stats_cache)
        finally:
            shutil_rmtree(sub)
        key = f"fid_{args.n_per_class}"
        row[key] = fid
        row[f"{key}_cfg"] = {"n_per_class": args.n_per_class,
                             "method": args.method, "seed": args.seed}
        json.dump(row, open(side, "w"), indent=2)
        old = row.get("fid")
        print(f"  {key}={fid:.4f}   (original fid={old})")


def collect(results_dir):
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "runs", "lam*_seed*")))
    grid, plot, n_tag = [], {}, None
    for run_dir in run_dirs:
        side = os.path.join(run_dir, "rows.json")
        if not os.path.exists(side):
            continue
        row = json.load(open(side))
        lam, seed = parse_lam_seed(run_dir)
        fid50 = row.get("fid")
        # figure out which fid_<N> key(s) exist; use most recent cfg if mixed
        keys = [k for k in row if k.startswith("fid_") and k.endswith("_cfg")]
        cfg = None
        for k in keys:
            cfg = row[k]  # take last written
        n = cfg.get("n_per_class") if cfg else None
        if n is None:
            continue
        fidn = row.get(f"fid_{n}")
        if fidn is None:
            continue
        if n_tag is None:
            n_tag = n
        grid.append({"lambda": lam, "seed": seed,
                     f"fid_{n}": fidn, "fid": fid50,
                     "diff": (fid50 - fidn) if isinstance(fid50, (int, float)) else None})
        plot.setdefault(lam, []).append(fidn)

    if not grid:
        raise SystemExit("no fid_<N> values found - run the compute step first")

    grid_csv = os.path.join(results_dir, f"fid{n_tag}_grid.csv")
    with open(grid_csv, "w") as f:
        f.write(f"lambda,seed,fid_{n_tag},fid,diff\n")
        for g in sorted(grid, key=lambda g: (g["lambda"], g["seed"])):
            f.write(f"{g['lambda']},{g['seed']},{g[f'fid_{n_tag}']},"
                    f"{g['fid'] if g['fid'] is not None else ''},"
                    f"{'' if g['diff'] is None else round(g['diff'], 4)}\n")

    plot_csv = os.path.join(results_dir, f"fid{n_tag}_plot.csv")
    with open(plot_csv, "w") as f:
        f.write("lambda,n_seeds,FID_mean,FID_std\n")
        for lam in sorted(plot):
            vals = sorted(plot[lam])
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            f.write(f"{lam},{len(vals)},{mean:.4f},{var ** 0.5:.4f}\n")

    print(f"wrote {grid_csv}")
    print(f"wrote {plot_csv}")


def np_load(path):
    import numpy as _np
    return _np.load(path)


def np_savez(path, **kw):
    import numpy as _np
    _np.savez(path, **kw)


def shutil_rmtree(path):
    import shutil as _s
    _s.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()