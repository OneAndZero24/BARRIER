#!/bin/bash -l
# ============================================================================
# SLURM Job — Flux NSFW Erasure + COCO 10K FID/CLIP (fixed subset)
# ============================================================================
# Parameters:
#   Model:      FLUX.1-dev
#   Method:     NSFW Erasure (InTAct)
#   Blocks:     15-18
#   k:          64 (reduced_dim)
#   lambda:     0.5
#   lr:         1e-3
#   Epochs:     5
#   Evaluation: NudeNet I2P + COCO 10K (deterministic FID & CLIP)
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_nsfw_10k.sh
# ============================================================================

#SBATCH --job-name=flux-nsfw-10k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# HuggingFace token
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: no Hugging Face token found; expected $HF_TOKEN_FILE or HF_TOKEN/HUGGINGFACE_HUB_TOKEN"
    exit 1
fi

# Cache configuration
if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
elif [ -w "/shared/results/common/miksa/intact/SD" ] || [ -w "/shared/results/common/miksa/intact/SD/.cache" ]; then
    CACHE_BASE="/shared/results/common/miksa/intact/SD/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

echo "============================================================"
echo "Flux NSFW Erasure + COCO 10K (fixed subset)"
echo "============================================================"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Config: configs/intact/pipeline_nsfw_10k.yaml"
echo "  Blocks: [15, 16, 17, 18]"
echo "  k (reduced_dim): 64"
echo "  lambda: 0.5"
echo "  lr: 1e-3"
echo "  epochs: 5"
echo "  COCO: 10K prompts, deterministic FID+CLIP"
echo "============================================================"

# Install torchmetrics if needed
if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torchmetrics") else 1)
PY
then
    echo "torchmetrics not found, installing..."
    python -m pip install --upgrade torchmetrics
fi

echo "=== Starting pipeline ==="
python intact_pipeline.py --config configs/intact/pipeline_nsfw_10k.yaml

echo "=== Pipeline complete ==="
