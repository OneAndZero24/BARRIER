#!/usr/bin/env python3
"""Generate a 95-image I2P gallery from a prompts CSV using Stable Diffusion.

Behavior:
- If the CSV has a column named 'nudity_percentage' the script prefers rows with >50.
- Otherwise it uses the first available text-like column or plain lines.
"""
import argparse
import csv
import os
import sys
import torch


def read_prompts(csv_path):
    # try CSV DictReader first
    prompts = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return []
            # prefer nudity_percentage > 50
            if 'nudity_percentage' in rows[0]:
                filtered = [r for r in rows if r.get('nudity_percentage')]
                try:
                    filtered = [r for r in rows if float(r.get('nudity_percentage', 0)) > 50.0]
                except Exception:
                    filtered = []
                src = filtered or rows
            else:
                src = rows

            # find a good text column
            text_col = None
            for key in ['prompt', 'text', 'prompt_text', 'caption']:
                if key in src[0]:
                    text_col = key
                    break
            if text_col:
                prompts = [r[text_col].strip() for r in src if r.get(text_col)]
            else:
                # fallback: use first non-empty column
                cols = list(src[0].keys())
                if cols:
                    prompts = [r[cols[0]].strip() for r in src if r.get(cols[0])]
    except Exception:
        # fallback: try reading as plain lines
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                prompts = [l.strip() for l in f if l.strip()]
        except Exception:
            prompts = []
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num', type=int, default=95)
    parser.add_argument('--base-model', default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    prompts = read_prompts(args.prompts_csv)
    if not prompts:
        print('No prompts found in', args.prompts_csv, file=sys.stderr)
        sys.exit(2)

    prompts = prompts[: args.num]

    try:
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        print('diffusers not installed or failed to import:', e, file=sys.stderr)
        sys.exit(3)

    device = args.device
    torch_dtype = torch.float16 if ('cuda' in device and torch.cuda.is_available()) else torch.float32
    # Disable diffusers safety checker to allow NSFW outputs (consistent with project NSFW pipeline)
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, safety_checker=None, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    for i, prompt in enumerate(prompts):
        seed = args.seed + i
        generator = torch.Generator(device=device if device.startswith('cuda') else 'cpu')
        try:
            generator = generator.manual_seed(seed)
        except Exception:
            generator = None

        out = pipe(prompt, guidance_scale=args.guidance, num_inference_steps=args.steps, generator=generator)
        image = out.images[0]
        fn = os.path.join(args.out_dir, f"{i:03d}.png")
        image.save(fn)
        print('Wrote', fn)


if __name__ == '__main__':
    main()
