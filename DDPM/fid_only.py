#!/usr/bin/env python3
"""
FID-only backfill for DDPM region-grid runs whose eval was killed before FID
was computed (rows with fid=""/nan, e.g. the old -+-+- console).  Uses the
ALREADY-SAVED fid_samples PNGs -- no training, no sampling.

FID backend (default ``tf``): the exact TensorFlow Inception-V3 evaluator from
evaluator.py (OpenAI classify_image_graph_def.pb graph, pool_3 2048-dim) --
the same code path and scale as the paper tables.  Requires tensorflow.
``torch_fidelity`` / ``torchmetrics-*`` are approximate torch alternatives
(different scale) for environments without tensorflow.

For each matching run dir the script patches <run_dir>/rows.json (fid,
fid_backend, + fid_note) via write_sidecar (ts bumped -> aggregation dedup
keeps the patched row).  Then run --summarize as usual.

Usage:
    python fid_only.py --results_dir /shared/results/common/miksa/intact/DDPM/r \
        --region_modes two_corner two_random --lambda 0.5 --dry_run
    python fid_only.py --results_dir ... --region_modes two_corner two_random --lambda 0.5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_CLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CLS_DIR, ".."))  # repo root (ablation_common)
sys.path.insert(0, _CLS_DIR)

from barrier.ablation_common import write_sidecar  # noqa: E402

log = logging.getLogger(__name__)

FID_DIR_PREFIX = "fid_samples_guidance_"
REF_DIR_DEFAULT = "/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"
CHUNK = 256


def _load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _imgs(folder, n):
    files = []
    for fmt in ("*.png", "*.jpg", "*.jpeg"):
        files += sorted(Path(folder).rglob(fmt))
    files = sorted(set(files))
    return files if not n or n <= 0 else files[:n]


def compute_fid_torchmetrics(ref_dir, fid_dir, n, device, note=None):
    import torch
    import torchmetrics
    import torchvision.transforms as T
    from PIL import Image

    t = T.Compose([T.ToTensor()])

    def _chunked(paths):
        for i in range(0, len(paths), CHUNK):
            imgs = torch.stack([t(Image.open(p).convert("RGB"))
                                for p in paths[i:i + CHUNK]])
            yield (imgs * 255).byte()

    fidm = torchmetrics.image.fid.FrechetInceptionDistance(
        feature=2048, reset_real_features=False).to(device)

    ref_paths = _imgs(ref_dir, n)
    gen_paths = _imgs(fid_dir, n)
    both = min(len(ref_paths), len(gen_paths))
    log.info(f"ref images: {len(ref_paths)}   gen images: {len(gen_paths)}"
             f"   using: {both}")
    if len(gen_paths) == 0:
        raise RuntimeError(f"no images found under {fid_dir}")
    if both < min(len(ref_paths), len(gen_paths)):
        log.warning("image sets differ in size; truncating to the smaller")
    t0 = time.time()
    for batch in _chunked(ref_paths[:both]):
        fidm.update(batch.to(device), real=True)
    for batch in _chunked(gen_paths[:both]):
        fidm.update(batch.to(device), real=False)
    fid = float(fidm.compute())
    log.info(f"FID = {fid:.3f}   ({time.time() - t0:.0f}s incl. inception)")
    return fid


def compute_fid_torch_fidelity(ref_dir, fid_dir, device, note=None):
    """Faithful torch port of pipeline.py's TF evaluator (Inception-v3 pool3,
    2048-dim, improving-renes weights, same preprocessing as evaluator.py)."""
    import torch_fidelity

    t0 = time.time()
    out = torch_fidelity.calculate_metrics(
        input1=ref_dir,
        input2=fid_dir,
        cuda=str(device).startswith("cuda"),
        fid=True,
        feature_layer=2048,
        verbose=False,
    )
    fid = float(out["frechet_inception_distance"])
    log.info(f"FID = {fid:.3f}   ({time.time() - t0:.0f}s incl. inception)")
    return fid


def compute_fid_tf(ref_dir, fid_dir, note=None):
    """EXACT table-scale FID: the TensorFlow Inception-V3 evaluator from
    evaluator.py (OpenAI guided-diffusion classify_image_graph_def.pb graph,
    pool_3 2048-dim) — the same code path that produced the paper tables.
    Requires tensorflow."""
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    log.info(f"ref: {ref_dir}")
    log.info(f"gen: {fid_dir}")
    ref_arr = read_images_folder(ref_dir)
    sample_arr = read_images_folder(fid_dir)
    log.info(f"ref images: {len(ref_arr)}   gen images: {len(sample_arr)}")

    t0 = time.time()
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
    log.info(f"FID = {fid:.4f}   ({time.time() - t0:.0f}s incl. inception)")
    return fid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="/shared/results/common/miksa/intact/DDPM/r")
    parser.add_argument("--region_modes", nargs="+",
                        default=["two_corner", "two_random"])
    parser.add_argument("--experiment", default=None,
                        help="only backfill runs with this experiment tag "
                             "(e.g. lambda_sweep); default = all experiments")
    parser.add_argument("--lambda", dest="lam", type=float, default=None,
                        help="only backfill this lambda; default = all lambdas")
    parser.add_argument("--n", type=int, default=0,
                        help="cap per-side image count (0 = use all)")
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT)
    parser.add_argument("--backend", choices=["tf",
                                              "torch_fidelity",
                                              "torchmetrics-2048",
                                              "torchmetrics-64"],
                        default="tf",
                        help="tf = the exact evaluator.py Inception graph (same "
                             "scale as the paper tables; needs tensorflow). "
                             "torch_fidelity / torchmetrics-* are approximate "
                             "torch alternatives (different scale).")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute even for runs that already have a FID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.isdir(args.ref_dir):
        log.error(f"reference dir missing: {args.ref_dir}")
        sys.exit(2)

    def _lam_matches(r, lam):
        if lam is None:
            return True
        try:
            return abs(float(r.get("lambda", float("nan"))) - lam) < 1e-9
        except (TypeError, ValueError):
            return str(r.get("lambda", "")) == str(lam)

    runs_root = os.path.join(args.results_dir, "runs")
    matches = []
    for run_dir in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, run_dir)
        side = _load(os.path.join(d, "rows.json"))
        if not side or "run" not in side:
            continue
        r = side["run"]
        if (r.get("region_mode") in args.region_modes
                and _lam_matches(r, args.lam)
                and (args.experiment is None
                     or r.get("experiment") == args.experiment)):
            matches.append(d)

    if not matches:
        log.error(f"no runs match region_modes={args.region_modes} "
                  f"lam={args.lam} experiment={args.experiment}")
        sys.exit(1)

    done = []
    for d in matches:
        side = _load(os.path.join(d, "rows.json"))
        r = side["run"]
        fid_dirs = sorted(Path(d).glob(f"{FID_DIR_PREFIX}*"))
        n_imgs = len(_imgs(fid_dirs[0], args.n)) if fid_dirs else 0
        old_fid = r.get("fid", "")
        has_fid = isinstance(old_fid, (int, float)) and old_fid == old_fid
        log.info(f"{Path(d).name}: region={r.get('region_mode')} "
                 f"lam={r.get('lambda')} seed={r.get('seed')} "
                 f"fid_dir={len(fid_dirs)} img={n_imgs} fid_old={old_fid!r}")
        if has_fid and not args.force:
            done.append((d, old_fid, r.get("fid_backend", "")))
            continue  # already has a numeric FID; leave it (use --force)
        if not fid_dirs:
            log.warning(f"no {FID_DIR_PREFIX}* dir in {d}; skipping")
            continue
        if args.dry_run:
            continue
        try:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            if args.backend == "tf":
                fid = compute_fid_tf(args.ref_dir, str(fid_dirs[0]))
            elif args.backend == "torch_fidelity":
                fid = compute_fid_torch_fidelity(args.ref_dir, str(fid_dirs[0]),
                                                 device)
            else:
                fid = compute_fid_torchmetrics(
                    args.ref_dir, str(fid_dirs[0]), args.n, device)
        except Exception as exc:
            log.warning(f"FID failed for {d}: {exc}")
            continue
        r["fid"] = fid
        r["fid_backend"] = args.backend
        r["fid_note"] = "backfill (" + {
            "tf": "exact evaluator.py Inception graph (table scale)",
            "torch_fidelity": "TF-evaluator-equivalent (improved-renes pool3)",
            "torchmetrics-2048": "torchmetrics pool3 2048-dim",
            "torchmetrics-64": "torchmetrics 64-dim (legacy table scale)",
        }[args.backend] + ")"
        write_sidecar(d, r, side.get("diagnostics") or [])
        done.append((d, fid, args.backend))

    print()
    if done:
        print(f"{'run_dir':<70s} {'fid':>10s}  backend")
        for d, fid, be in done:
            print(f"{Path(d).name:<70s} {fid:>10.3f}  {be}")
    else:
        print("nothing to do (dry run or all already have FID)")


if __name__ == "__main__":
    main()