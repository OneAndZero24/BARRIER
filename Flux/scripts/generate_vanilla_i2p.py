#!/usr/bin/env python3
"""
Generate vanilla Flux I2P images using the same settings as the InTAct pipeline.
Loads sampling settings from a pipeline config YAML and calls eval.generate_images.

Usage:
    python scripts/generate_vanilla_i2p.py --config configs/intact/pipeline_nsfw.yaml
"""
import sys
from pathlib import Path
import argparse
import yaml

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from eval.generate_images import generate_images

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "intact" / "pipeline_nsfw.yaml"


def load_settings(config_path: Path):
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    pc = cfg.get('paths', {})
    uc = cfg.get('unlearn', {})
    ec = cfg.get('evaluate', {})

    settings = {}
    settings['prompts_path'] = pc.get('prompts', '../SD/prompts/unsafe-prompts4703.csv')
    settings['output_dir'] = pc.get('output_dir', 'results/vanilla_i2p')
    settings['device'] = f"cuda:{cfg.get('pipeline', {}).get('device', '0')}"
    settings['guidance_scale'] = float(ec.get('guidance_scale', 3.5))
    settings['ddim_steps'] = int(ec.get('ddim_steps', 28))
    settings['image_size'] = int(uc.get('resolution', 512))
    settings['num_samples'] = int(ec.get('num_samples_per_prompt', 1))
    settings['batch_size'] = int(ec.get('generation_batch_size', 8))
    settings['base_model_path'] = str(uc.get('pretrained_model_name_or_path', 'black-forest-labs/FLUX.1-dev'))
    settings['model_dir'] = pc.get('model_save_dir', 'models')
    settings['max_prompts'] = ec.get('max_prompts', None)
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--prompts', default=None, help='Override prompts CSV path')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--base-model-path', default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    settings = load_settings(cfg_path)

    prompts_path = args.prompts or settings['prompts_path']
    output_dir = args.output_dir or settings['output_dir']
    device = args.device or settings['device']
    steps = args.steps if args.steps is not None else settings['ddim_steps']
    guidance = args.guidance_scale if args.guidance_scale is not None else settings['guidance_scale']
    image_size = args.image_size if args.image_size is not None else settings['image_size']
    num_samples = args.num_samples if args.num_samples is not None else settings['num_samples']
    batch_size = args.batch_size if args.batch_size is not None else settings['batch_size']
    base_model = args.base_model_path if args.base_model_path is not None else settings['base_model_path']
    model_dir = settings['model_dir']
    max_prompts = settings['max_prompts']

    print("Generating vanilla I2P images with settings:")
    print(f"  prompts: {prompts_path}")
    print(f"  output: {output_dir}")
    print(f"  device: {device}")
    print(f"  steps: {steps}")
    print(f"  guidance: {guidance}")
    print(f"  image size: {image_size}")
    print(f"  num samples/prompt: {num_samples}")
    print(f"  batch size: {batch_size}")
    print(f"  base model: {base_model}")

    generate_images(
        model_name="",
        prompts_path=prompts_path,
        save_path=output_dir,
        device=device,
        guidance_scale=guidance,
        image_size=image_size,
        ddim_steps=steps,
        num_samples=num_samples,
        from_case=0,
        base_model_path=base_model,
        model_dir=model_dir,
        max_prompts=max_prompts,
        batch_size=batch_size,
    )


if __name__ == '__main__':
    main()
