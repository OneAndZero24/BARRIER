#!/usr/bin/env python3
"""
Compute COCO FID and CLIP metrics from pregenerated images.

Minimal standalone script: no model loading, no generation, only metrics.
Installs torchmetrics if missing.

Usage:
  python eval_coco_metrics_only.py \\
    --images-dir /path/to/pregenerated/images \\
    --prompts-csv /path/to/coco_prompts.csv \\
    [--device cuda:0] \\
    [--image-size 512] \\
    [--fid-feature 2048] \\
    [--max-real 5000] \\
    [--max-fake 5000] \\
    [--coco-images-dir /path/to/real/coco/val/images] \\
    [--coco-ann-path /path/to/coco_captions_val2014.json]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure torchmetrics is installed before importing evaluate modules
def ensure_torchmetrics():
    """Install torchmetrics and torch-fidelity if not available."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("torchmetrics.image.fid")
        if spec is not None:
            return  # Already installed
    except (ImportError, AttributeError):
        pass
    
    print("torchmetrics or torch-fidelity not found, installing...")
    
    # Install both torch-fidelity (required for FID) and torchmetrics
    packages = ["torch-fidelity>=0.3.0", "torchmetrics[image]>=1.0"]
    for pkg in packages:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"WARNING: Failed to install {pkg}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
    
    # Verify FID is now importable
    try:
        import importlib.util
        spec = importlib.util.find_spec("torchmetrics.image.fid")
        if spec is None:
            print("ERROR: torchmetrics.image.fid still not found after installation")
            sys.exit(1)
    except (ImportError, AttributeError) as e:
        print(f"ERROR: Cannot import torchmetrics.image.fid even after installation: {e}")
        sys.exit(1)
    
    print("torch-fidelity and torchmetrics installed successfully")

ensure_torchmetrics()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FLUX_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLUX_ROOT))

import setup_cache  # noqa: F401,E402

from eval.evaluate import compute_clip_score_coco, compute_fid_coco  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute COCO FID and CLIP metrics from pregenerated images"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing pregenerated COCO images",
    )
    parser.add_argument(
        "--prompts-csv",
        required=True,
        help="Path to CSV with case_number and prompt columns",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device string (default: cuda:0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size for FID computation (default: 512)",
    )
    parser.add_argument(
        "--fid-feature",
        type=int,
        default=2048,
        help="InceptionV3 feature dimension for FID (default: 2048)",
    )
    parser.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Max real images to use (default: all)",
    )
    parser.add_argument(
        "--max-fake",
        type=int,
        default=None,
        help="Max fake images to use (default: all)",
    )
    parser.add_argument(
        "--coco-images-dir",
        default=None,
        help="Directory with real COCO val images (if None, attempts HF download)",
    )
    parser.add_argument(
        "--coco-ann-path",
        default=None,
        help="Path to COCO captions JSON (if None, attempts HF download)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.images_dir):
        print(f"ERROR: images directory not found: {args.images_dir}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.prompts_csv):
        print(f"ERROR: prompts CSV not found: {args.prompts_csv}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("COCO Metrics Evaluation (FID + CLIP)")
    print("=" * 70)
    print(f"Images dir:     {args.images_dir}")
    print(f"Prompts CSV:    {args.prompts_csv}")
    print(f"Device:         {args.device}")
    print(f"Image size:     {args.image_size}")
    print(f"FID feature:    {args.fid_feature}")
    if args.coco_images_dir:
        print(f"Real COCO imgs: {args.coco_images_dir}")
    if args.coco_ann_path:
        print(f"COCO ann path:  {args.coco_ann_path}")
    print()

    # Compute FID
    print("Computing FID...")
    fid_score = compute_fid_coco(
        args.images_dir,
        coco_images_dir=args.coco_images_dir,
        coco_ann_path=args.coco_ann_path,
        image_size=args.image_size,
        feature=args.fid_feature,
        max_real=args.max_real,
        max_fake=args.max_fake,
    )

    # Compute CLIP
    print("Computing CLIP score...")
    clip_score = compute_clip_score_coco(args.images_dir, args.prompts_csv, args.device)

    # Summary
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"FID_COCO:  {fid_score}")
    print(f"CLIP_COCO: {clip_score}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
