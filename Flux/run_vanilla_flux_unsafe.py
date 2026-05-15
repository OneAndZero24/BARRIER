#!/usr/bin/env python3
"""
Run vanilla Flux (no unlearning) on specific unsafe prompts.
Extracts prompts from unsafe-prompts4703.csv and generates images with specified seeds/guidance.

Usage:
    python run_vanilla_flux_unsafe.py
"""
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import FluxPipeline
from PIL import Image

# Specific case numbers to run
TARGET_CASES = [296, 327, 649, 698, 1066, 1276, 1308]


def load_prompts_from_csv(csv_path: str, case_numbers: List[int]) -> List[Dict]:
    """
    Load prompts from CSV file filtered by case_number.
    
    Returns list of dicts with keys: case_number, prompt, evaluation_seed, evaluation_guidance, 
                                    sd_image_width, sd_image_height
    """
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_num = int(row['case_number'])
            if case_num in case_numbers:
                prompts.append({
                    'case_number': case_num,
                    'prompt': row['prompt'],
                    'seed': int(row['evaluation_seed']),
                    'guidance': float(row['evaluation_guidance']),
                    'width': int(row['sd_image_width']),
                    'height': int(row['sd_image_height']),
                })
    
    # Sort by case_number for consistent ordering
    prompts.sort(key=lambda x: x['case_number'])
    return prompts


def setup_pipeline(device: str = "cuda:0", dtype = torch.bfloat16):
    """
    Initialize vanilla Flux pipeline (no LoRA, no unlearning).
    """
    print(f"Loading Flux.1-dev model to {device}...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    return pipe


def generate_images(
    pipe: FluxPipeline,
    prompts: List[Dict],
    output_dir: str = "results/vanilla_flux",
    num_inference_steps: int = 28,
):
    """
    Generate images for all prompts using the pipeline.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(prompts)} images to {output_dir}\n")
    
    results = []
    for prompt_data in prompts:
        case_num = prompt_data['case_number']
        prompt = prompt_data['prompt']
        seed = prompt_data['seed']
        guidance = prompt_data['guidance']
        width = prompt_data['width']
        height = prompt_data['height']
        
        print(f"[{case_num}] Generating with seed={seed}, guidance={guidance}")
        print(f"     Prompt: {prompt[:80]}...")
        
        try:
            # Set generator with seed for reproducibility
            generator = torch.Generator("cuda").manual_seed(seed)
            
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance,
                generator=generator,
                max_sequence_length=256,
            ).images[0]
            
            # Save with case number
            output_file = output_path / f"case_{case_num:04d}.png"
            image.save(str(output_file))
            print(f"     ✓ Saved to {output_file}\n")
            
            results.append({
                'case_number': case_num,
                'prompt': prompt,
                'seed': seed,
                'guidance': guidance,
                'output': str(output_file),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"     ✗ Error: {e}\n")
            results.append({
                'case_number': case_num,
                'prompt': prompt,
                'seed': seed,
                'guidance': guidance,
                'output': None,
                'status': f'failed: {str(e)}'
            })
    
    return results


def save_results_summary(results: List[Dict], output_dir: str):
    """
    Save a summary of all results to a CSV file.
    """
    summary_path = Path(output_dir) / "generation_summary.csv"
    
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_number', 'seed', 'guidance', 'status', 'prompt', 'output'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nSummary saved to {summary_path}")
    
    # Print summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    print(f"Results: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description='Run vanilla Flux on unsafe prompts')
    parser.add_argument('--csv', type=str, default='../SD/prompts/unsafe-prompts4703.csv',
                        help='Path to unsafe prompts CSV file')
    parser.add_argument('--output-dir', type=str, default='results/vanilla_flux',
                        help='Output directory for generated images')
    parser.add_argument('--steps', type=int, default=28,
                        help='Number of inference steps')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (cuda:0, cuda:1, etc.)')
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path
    
    print("=" * 80)
    print("VANILLA FLUX INFERENCE ON UNSAFE PROMPTS")
    print("=" * 80)
    
    # Load prompts
    print(f"\nLoading prompts from: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    prompts = load_prompts_from_csv(str(csv_path), TARGET_CASES)
    print(f"Loaded {len(prompts)} prompts (case numbers: {[p['case_number'] for p in prompts]})")
    
    # Setup pipeline
    pipe = setup_pipeline(device=args.device)
    
    # Generate images
    results = generate_images(
        pipe,
        prompts,
        output_dir=args.output_dir,
        num_inference_steps=args.steps,
    )
    
    # Save summary
    save_results_summary(results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
