#!/usr/bin/env python3
"""
FID-only backfill for DDPM region-grid runs whose eval was killed before FID
was computed (rows with fid=""/nan, e.g. the old -+-+- console).  Uses the
ALREADY-SAVED fid_samples PNGs -- no training, no sampling, no tensorflow.

FID backend: torchmetrics FrechetInceptionDistance(feature=2048) -- Inception
V3 pool3 activations, the same 2048-dim mu/sigma statistics as the paper's TF
evaluator.  (The old fallback used feature=64, a different scale; keep 2048.)

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
sys.path.insert(0, os.path.join(_CLS_DIR, "..", ".."))  # repo root
sys.path.insert(0, _CLS_DIR)

from ablation_common import write_sidecar  # noqa: E402

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
    return files[:n]


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
    log.info(f"ref images: {len(ref_paths)}   gen images: {len(gen_paths)}"
             f"   n={n}")
    if len(gen_paths) == 0:
        raise RuntimeError(f"no images found under {fid_dir}")
    both = max(len(ref_paths), len(gen_paths))
    if min(len(ref_paths), len(gen_paths)) < n:
        log.warning("fewer than requested images on disk; using what's there")
    t0 = time.time()
    for batch in _chunked(ref_paths[:both]):
        fidm.update(batch, real=True)
    for batch in _chunked(gen_paths[:both]):
        fidm.update(batch, real=False)
    fid = float(fidm.compute())
    log.info(f"FID = {fid:.3f}   ({time.time() - t0:.0f}s incl. inception)")
    return fid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="/shared/results/common/miksa/intact/DDPM/r")
    parser.add_argument("--region_modes", nargs="+",
                        default=["two_corner", "two_random"])
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.isdir(args.ref_dir):
        log.error(f"reference dir missing: {args.ref_dir}")
        sys.exit(2)

    runs_root = os.path.join(args.results_dir, "runs")
    matches = []
    for run_dir in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, run_dir)
        side = _load(os.path.join(d, "rows.json"))
        if not side or "run" not in side:
            continue
        r = side["run"]
        if (r.get("region_mode") in args.region_modes
                and str(r.get("lambda", "")) == str(args.lam)):
            matches.append(d)

    if not matches:
        log.error(f"no runs match region_modes={args.region_modes} lam={args.lam}")
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
        if has_fid:
            done.append((d, old_fid, r.get("fid_backend", "")))
            continue  # already has a numeric FID; leave it
        if not fid_dirs:
            log.warning(f"no {FID_DIR_PREFIX}* dir in {d}; skipping")
            continue
        if args.dry_run:
            continue
        try:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            fid = compute_fid_torchmetrics(args.ref_dir, str(fid_dirs[0]),
                                           args.n, device)
        except Exception as exc:
            log.warning(f"FID failed for {d}: {exc}")
            continue
        r["fid"] = fid
        r["fid_backend"] = "torchmetrics-2048"
        r["fid_note"] = "backfill (pool3 2048-dim, TF-equivalent stats)"
        write_sidecar(d, r, side.get("diagnostics") or [])
        done.append((d, fid, "torchmetrics-2048"))

    print()
    if done:
        print(f"{'run_dir':<70s} {'fid':>10s}  backend")
        for d, fid, be in done:
            print(f"{Path(d).name:<70s} {fid:>10.3f}  {be}")
    else:
        print("nothing to do (dry run or all already have FID)")


if __name__ == "__main__":
    main()