#!/usr/bin/env python3
"""
Environment verification script for SD NSFW ablations.
Run this before submitting SLURM jobs to ensure everything is set up correctly.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    print("Python version:", sys.version)
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print("✓ Python version OK")


def check_imports():
    print("\n=== Checking imports ===")
    
    # Core
    import torch
    import numpy as np
    import yaml
    import pandas as pd
    import matplotlib
    print("✓ Core libs (torch, numpy, yaml, pandas, matplotlib)")
    
    # Diffusers / transformers
    import diffusers
    import transformers
    print(f"✓ diffusers {diffusers.__version__}, transformers {transformers.__version__}")
    
    # WandB
    import wandb
    print(f"✓ wandb {wandb.__version__}")
    
    # NudeNet
    try:
        import nudenet
        print("✓ nudenet")
    except ImportError:
        print("⚠ nudenet not installed (needed for evaluation)")
    
    # CLIP
    import clip
    print("✓ clip")
    
    # Torchmetrics for FID
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        print("✓ torchmetrics FID")
    except ImportError:
        print("⚠ torchmetrics not installed (needed for FID)")
    
    # SD-specific
    try:
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf
        print("✓ ldm, omegaconf")
    except ImportError as e:
        print(f"✗ ldm/omegaconf import failed: {e}")
        raise


def check_cuda():
    print("\n=== Checking CUDA ===")
    import torch
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    # Quick tensor test
    x = torch.randn(2, 2).cuda()
    y = x @ x.T
    print(f"  GPU compute test: OK (result sum={y.sum().item():.4f})")
    return True


def check_paths():
    print("\n=== Checking paths ===")
    scratch = os.environ.get("SCRATCH", "/net/scratch/hscra/plgrid/plgmiksa")
    home = os.environ.get("HOME", "/net/home/plgrid/plgmiksa")
    
    checks = [
        ("SCRATCH env", scratch, True),
        ("HOME env", home, True),
        ("Repo", f"{home}/InTAct-Unl/SD", True),
        ("Checkpoint", f"{scratch}/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt", True),
        ("NSFW data", f"{scratch}/SD/data/nsfw", True),
        ("NOT_NSFW data", f"{scratch}/SD/data/not-nsfw", True),
        ("Config (fulleval)", f"{home}/InTAct-Unl/SD/configs/pipeline_nsfw_fulleval.yaml", True),
        ("SD config", f"{home}/InTAct-Unl/SD/configs/stable-diffusion/v1-intact.yaml", True),
        ("Diffusers config", f"{home}/InTAct-Unl/SD/diffusers_unet_config.json", True),
        ("NSFW prompts", f"{home}/InTAct-Unl/SD/prompts/unsafe-prompts4703.csv", True),
        ("ImageNet prompts", f"{home}/InTAct-Unl/SD/prompts/imagenette.csv", True),
        ("Venv activate", f"{scratch}/sd_venv/bin/activate", True),
    ]
    
    all_ok = True
    for name, path, required in checks:
        exists = Path(path).exists()
        status = "✓" if exists else ("✗" if required else "⚠")
        print(f"  {status} {name}: {path}")
        if required and not exists:
            all_ok = False
    
    return all_ok


def check_venv():
    print("\n=== Checking venv ===")
    venv_python = Path(os.environ.get("SCRATCH", "/net/scratch/hscra/plgrid/plgmiksa")) / "sd_venv/bin/python"
    if not venv_python.exists():
        print(f"✗ Venv python not found at {venv_python}")
        return False
    
    # Check key packages in venv
    result = subprocess.run(
        [str(venv_python), "-c", "import torch; print(torch.__version__); import diffusers; print(diffusers.__version__)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✓ Venv works: {result.stdout.strip()}")
    else:
        print(f"✗ Venv check failed: {result.stderr}")
        return False
    return True


def check_pipeline_load():
    print("\n=== Checking pipeline config load ===")
    sys.path.insert(0, str(Path(os.environ.get("HOME", "/net/home/plgrid/plgmiksa")) / "InTAct-Unl"))
    
    import yaml
    config_path = Path(os.environ.get("HOME", "/net/home/plgrid/plgmiksa")) / "InTAct-Unl/SD/configs/pipeline_nsfw_fulleval.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"✓ Config loaded: {config_path}")
        print(f"  unlearn.lr = {cfg['unlearn']['lr']}")
        print(f"  intact.lambda = {cfg['intact']['lambda_interval']}")
        print(f"  intact.targets = {cfg['intact']['targets']}")
    except Exception as e:
        print(f"✗ Config load failed: {e}")
        return False
    return True


def check_slurm_scripts():
    print("\n=== Checking SLURM scripts ===")
    home = os.environ.get("HOME", "/net/home/plgrid/plgmiksa")
    scripts = [
        "slurm_sd_nsfw_reduced_dim_ablation.sh",
        "slurm_sd_nsfw_ablation_forget_fraction.sh",
        "slurm_sd_nsfw_ablation_remain_fraction.sh",
    ]
    all_ok = True
    for script in scripts:
        path = Path(home) / "InTAct-Unl/SD/scripts" / script
        if path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} not found at {path}")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("SD NSFW Ablation Environment Check")
    print("=" * 60)
    
    checks = [
        ("Python version", check_python_version),
        ("Imports", check_imports),
        ("CUDA", check_cuda),
        ("Paths", check_paths),
        ("Venv", check_venv),
        ("Pipeline config", check_pipeline_load),
        ("SLURM scripts", check_slurm_scripts),
    ]
    
    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ✗ {name} failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
    
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()