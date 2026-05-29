#!/usr/bin/env python3
"""Generate a 95-image gallery using the repo's SD eval pipeline.

This matches the normal SD NSFW flow because it delegates to
`eval-scripts/generate-images.py`, which reads prompts from a CSV and does not
apply a diffusers safety checker that can turn outputs black.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num", type=int, default=95)
    parser.add_argument("--base-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.prompts_csv)
    if "prompt" not in df.columns:
        raise SystemExit(f"Expected a 'prompt' column in {args.prompts_csv}")

    if "nudity_percentage" not in df.columns:
        raise SystemExit(f"Expected a 'nudity_percentage' column in {args.prompts_csv}")

    df = df[df["nudity_percentage"] > 0.5].copy()
    if df.empty:
        raise SystemExit(f"No prompts left after nudity_percentage > 0.5 filter in {args.prompts_csv}")

    eval_scripts_dir = Path(__file__).resolve().parents[2] / "eval-scripts" / "generate-images.py"
    spec = importlib.util.spec_from_file_location("generate_images", eval_scripts_dir)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    tokenizer = None
    try:
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    except Exception:
        tokenizer = None

    if tokenizer is not None:
        token_lengths = df["prompt"].astype(str).map(
            lambda prompt: len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors=None,
                ).input_ids[0]
            )
        )
        df = df[token_lengths <= 77].copy()
        if df.empty:
            raise SystemExit(f"No prompts left after token-length <= 77 filter in {args.prompts_csv}")

    if len(df) < args.num:
        raise SystemExit(
            f"Only {len(df)} prompts remain after filtering, fewer than requested --num {args.num}."
        )

    filtered_prompts_csv = out_dir / "i2p_prompts_filtered.csv"
    df.to_csv(filtered_prompts_csv, index=False)

    module.generate_images(
        model_name="",
        prompts_path=str(filtered_prompts_csv),
        save_path=str(out_dir),
        device=args.device,
        guidance_scale=args.guidance,
        image_size=512,
        ddim_steps=args.steps,
        num_samples=1,
        from_case=0,
        base_model_path=args.base_model,
        base_config_path=None,
        model_dir=args.model_dir,
        max_prompts=args.num,
        n_outer=1,
    )


if __name__ == "__main__":
    main()
