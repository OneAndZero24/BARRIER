"""
Redirect all model / dataset caches to a shared scratch directory so that
the home-directory disk quota is not exceeded.

Import this module **before** any library that downloads models (torch,
transformers, diffusers, datasets, clip, wandb, …).

    import setup_cache          # sets env vars once
    import torch, transformers  # libraries now use the redirected paths

The module is idempotent – re-importing is a no-op.

Override the default root by setting ``CACHE_ROOT`` in the environment
*before* importing this module.
"""

import os

def _resolve_cache_root() -> str:
    """Resolve cache root from config/env/auto-detection, prioritizing scratch."""
    candidates = []

    # 1. Explicit config from environment
    if os.environ.get("CACHE_ROOT"):
        candidates.append(os.environ["CACHE_ROOT"])

    # 2. SCRATCH environment variable (e.g., on HPC clusters)
    if os.environ.get("SCRATCH"):
        candidates.append(os.path.join(os.environ["SCRATCH"], ".cache"))
    
    # 3. Auto-detect cluster scratch from home path
    # On PLGrid: /net/home/plgrid/USERNAME → /net/scratch/hscra/plgrid/USERNAME
    home = os.path.expanduser("~")
    if "/net/home/plgrid/" in home:
        username = home.split("/")[-1]
        plgrid_scratch = f"/net/scratch/hscra/plgrid/{username}/.cache"
        candidates.append(plgrid_scratch)
    
    if os.environ.get("XDG_CACHE_HOME"):
        candidates.append(os.environ["XDG_CACHE_HOME"])

    # Legacy setups as fallback only (not scratch)
    candidates.append("/shared/results/common/miksa/intact/SD/.cache")

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"[setup_cache] Using cache root: {path}")
            return path
        except OSError:
            continue

    raise RuntimeError(
        f"Failed to find writable cache root. Tried: {candidates}"
    )


_CACHE_ROOT = _resolve_cache_root()

# HuggingFace (hub, datasets, tokenizers, diffusers)
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))

# PyTorch hub / torchvision pretrained weights / torch-fidelity
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE_ROOT, "torch"))

# General XDG cache (used by SD/ldm/data/imagenet.py for autoencoders)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)

# WandB run files
os.environ.setdefault("WANDB_DIR", os.path.join(_CACHE_ROOT, "wandb"))
os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(_CACHE_ROOT, "wandb"))

# OpenAI CLIP (does not respect XDG_CACHE_HOME, needs its own variable)
# The clip library checks download_root; we patch via env so callers
# can use:  clip.load(..., download_root=os.environ.get("CLIP_CACHE_DIR"))
os.environ.setdefault("CLIP_CACHE_DIR", os.path.join(_CACHE_ROOT, "clip"))

# Ensure the top-level cache directory exists
os.makedirs(_CACHE_ROOT, exist_ok=True)
