#!/usr/bin/env python3
"""
FID backfill for the ddpm-fid-repro lambda sweep, using the ORIGINAL FID
setup from the initial public release: the TensorFlow Inception-V3 graph in
evaluator.py (pool_3 2048-dim, spatial mixed_6/conv) through the exact code
path of compute_fid_full.py.

Scans <results_dir>/runs/*/rows.json and patches any run whose sidecar has
no numeric FID (e.g. the pipeline skipped the inline FID step), computing it
from the already-saved fid_samples_* PNGs vs the reference dataset
(5000 images/class, minus the forgotten class 0).

Requires tensorflow (present in salun-ddpm; see requirements.txt).

Usage:
    cd DDPM
    python scripts/fid_backfill_lambda_sweep.py \
        --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
    # dry run:
    python scripts/fid_backfill_lambda_sweep.py --results_dir ... --dry_run
"""

import argparse
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) or ".")

REF_DIR_DEFAULT = "/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"
SAMPLE_GLOBS = (
    "fid_samples_guidance_*_excluded_class_*",
    "fid_samples_without_label_*_guidance_*",
    "fid_samples",
)


def find_fid_sample_dir(run_dir):
    for ts_dir in sorted(glob.glob(os.path.join(run_dir, "output", "*"))):
        if not os.path.isdir(ts_dir):
            continue
        for pat in SAMPLE_GLOBS:
            hits = sorted(glob.glob(os.path.join(ts_dir, pat)))
            if hits and os.path.isdir(hits[0]) and os.listdir(hits[0]):
                return hits[0]
    return None


def compute_fid(ref_dir, sample_dir):
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    ref_arr = read_images_folder(ref_dir)
    sample_arr = read_images_folder(sample_dir)
    print(f"  reference={len(ref_arr)}  samples={len(sample_arr)}")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    evaluator = Evaluator(sess)
    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_arr)
    ref_stats, _ = evaluator.read_statistics(ref_acts)
    sample_acts = evaluator.read_activations(sample_arr)
    sample_stats, _ = evaluator.read_statistics(sample_acts)

    fid = float(sample_stats.frechet_distance(ref_stats))
    sess.close()
    return fid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if rows.json already has FID")
    args = parser.parse_args()

    if not os.path.isdir(args.ref_dir):
        raise SystemExit(f"reference dataset not found: {args.ref_dir}\n"
                         f"create it first:  python save_base_dataset.py "
                         f"--dataset cifar10 --label_to_forget 0")

    rows = sorted(glob.glob(os.path.join(args.results_dir, "runs", "*", "rows.json")))
    if not rows:
        raise SystemExit(f"no rows.json under {args.results_dir}/runs/")

    todo = []
    for p in rows:
        r = json.load(open(p))
        fid = r.get("fid")
        has_fid = isinstance(fid, (int, float)) and fid == fid
        if has_fid and not args.force:
            continue
        todo.append((p, r))

    print(f"{len(todo)}/{len(rows)} runs need FID backfill")
    if args.dry_run:
        for p, r in todo:
            print(f"  would backfill {p}  (lam={r.get('lambda')} seed={r.get('seed')})")
        return

    for p, r in todo:
        run_dir = os.path.dirname(p)
        sample_dir = find_fid_sample_dir(run_dir)
        if sample_dir is None:
            print(f"SKIP {p}: no fid_samples dir found under {run_dir}/output/*/")
            continue
        print(f"FID backfill {p}")
        print(f"  samples: {sample_dir}")
        fid = compute_fid(args.ref_dir, sample_dir)
        r["fid"] = fid
        r["fid_source"] = "evaluator.py tf inception (compute_fid_full path)"
        with open(p, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  -> fid={fid:.4f}  patched {p}")

    print("done")


if __name__ == "__main__":
    main()