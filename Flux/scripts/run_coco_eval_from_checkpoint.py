#!/usr/bin/env python3
"""Run COCO-only evaluation for a saved Flux checkpoint.

This script loads a fine-tuned Flux transformer checkpoint directly from a
`.safetensors` path, generates MS-COCO captioned samples or reuses
pregenerated images, and computes only:
  - COCO FID
  - COCO CLIP score

The defaults are tuned for the checkpoint path shown in the run log, but all
paths and evaluation sizes can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FLUX_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))  # for setup_cache
sys.path.insert(0, str(FLUX_ROOT))   # for eval modules

import setup_cache  # noqa: F401,E402  # ensure cache env is configured early

from eval.evaluate import compute_clip_score_coco, compute_fid_coco, generate_coco_prompts_csv  # noqa: E402
from eval.generate_images import generate_images  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/net/scratch/hscra/plgrid/plgmiksa/models/"
    "flux-intact-nsfw-nude-targets_blk15-16-17-18_proj_proj-"
    "lambda_0.5-epochs_5-lr_5e-05.safetensors"
)

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_PROMPTS = str(REPO_ROOT / "SD" / "prompts" / "coco_30k.csv")


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {key: expand_env_vars(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return expand_env_vars(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only COCO FID/CLIP for a Flux checkpoint")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "intact" / "pipeline_nsfw.yaml"),
        help="Flux YAML config used for base model, image size, and COCO paths",
    )
    parser.add_argument(
        "--model-weights-path",
        default=DEFAULT_CHECKPOINT,
        help="Path to the fine-tuned Flux transformer weights (.safetensors)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base Flux model used to build the pipeline before loading the checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where generated COCO images and prompts CSV will be written",
    )
    parser.add_argument(
        "--pregenerated-images-dir",
        default=None,
        help="Directory containing already-generated COCO images; skips image generation when set",
    )
    parser.add_argument(
        "--coco-prompts-csv",
        default=None,
        help="Optional local COCO prompts CSV; defaults to the config value or SD/prompts/coco_30k.csv",
    )
    parser.add_argument(
        "--coco-ann-path",
        default=None,
        help="Optional COCO annotations JSON for real-image FID; if omitted, HuggingFace fallback is used",
    )
    parser.add_argument(
        "--coco-images-dir",
        default=None,
        help="Optional directory containing real COCO validation images for FID",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device string, for example cuda:0. Defaults to the config device.",
    )
    parser.add_argument("--n-captions", type=int, default=10000)
    parser.add_argument("--num-samples-per-prompt", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_cfg = cfg.get("evaluate", {})
    paths_cfg = cfg.get("paths", {})
    unlearn_cfg = cfg.get("unlearn", {})

    device = args.device or f"cuda:{cfg.get('pipeline', {}).get('device', '0')}"
    output_dir = args.output_dir or paths_cfg.get("output_dir", str(Path("evaluation")))
    os.makedirs(output_dir, exist_ok=True)

    coco_prompts_csv = (
        args.coco_prompts_csv
        or evaluate_cfg.get("coco", {}).get("pregenerated_prompts_csv")
        or paths_cfg.get("coco_prompts_csv")
        or DEFAULT_PROMPTS
    )
    coco_ann_path = args.coco_ann_path if args.coco_ann_path is not None else paths_cfg.get("coco_ann_path")
    coco_images_dir = args.coco_images_dir if args.coco_images_dir is not None else paths_cfg.get("coco_images_dir")

    model_weights_path = args.model_weights_path
    model_name = Path(model_weights_path).stem

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else evaluate_cfg.get("guidance_scale", 3.5)
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else evaluate_cfg.get("ddim_steps", unlearn_cfg.get("ddim_steps", 28))
    image_size = args.image_size if args.image_size is not None else unlearn_cfg.get("resolution", 512)

    coco_eval_dir = os.path.join(output_dir, "coco_eval")
    os.makedirs(coco_eval_dir, exist_ok=True)
    coco_prompts_out = os.path.join(coco_eval_dir, "coco_prompts.csv")
    pregenerated_images_dir = args.pregenerated_images_dir

    if pregenerated_images_dir and not os.path.isdir(pregenerated_images_dir):
        raise FileNotFoundError(f"Pregenerated images directory not found: {pregenerated_images_dir}")

    print(f"Loading checkpoint: {model_weights_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {coco_eval_dir}")
    print(f"COCO prompts source: {coco_prompts_csv}")
    if pregenerated_images_dir:
        print(f"Pregenerated images dir: {pregenerated_images_dir}")

    generate_coco_prompts_csv(
        coco_prompts_out,
        n=args.n_captions,
        coco_ann_path=coco_ann_path,
        coco_prompts_csv_path=coco_prompts_csv,
    )

    if pregenerated_images_dir:
        coco_gen_dir = pregenerated_images_dir
    else:
        coco_gen_dir = generate_images(
            model_name=model_name,
            prompts_path=coco_prompts_out,
            save_path=coco_eval_dir,
            device=device,
            guidance_scale=guidance_scale,
            image_size=image_size,
            ddim_steps=ddim_steps,
            num_samples=args.num_samples_per_prompt,
            base_model_path=args.base_model,
            model_dir=os.path.dirname(model_weights_path),
            model_weights_path=model_weights_path,
            max_prompts=args.max_prompts,
            batch_size=args.generation_batch_size,
            strict_model_weights=True,
        )

    fid_score = compute_fid_coco(
        coco_gen_dir,
        coco_images_dir=coco_images_dir,
        coco_ann_path=coco_ann_path,
        image_size=image_size,
        n=args.n_captions,
        feature=args.fid_feature,
        max_real=args.max_real,
        max_fake=args.max_fake,
    )
    clip_score = compute_clip_score_coco(coco_gen_dir, coco_prompts_out, device)

    print("\n=== COCO evaluation summary ===")
    print(f"Checkpoint: {model_weights_path}")
    print(f"Generated images: {coco_gen_dir}")
    print(f"FID_COCO: {fid_score}")
    print(f"CLIP_COCO: {clip_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())