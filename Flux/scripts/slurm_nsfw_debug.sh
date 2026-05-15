#!/bin/bash -l
# ============================================================================
# SLURM Job – Flux NSFW Debug Run (single run, no sweep)
# ============================================================================
# Purpose:
# - Quick end-to-end validation before launching full sweeps
# - Runs a single config directly (no wandb agent/sweep)
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_nsfw_debug.sh
#
# Optional overrides:
#   sbatch --export=CONFIG_PATH=configs/intact/pipeline_nsfw.yaml,USE_WANDB=1 scripts/slurm_nsfw_debug.sh
# ============================================================================

#SBATCH --job-name=flux-nsfw-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# ---- Environment ----
ml ML-bundle/24.06a
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Load Hugging Face credentials from the saved token file if present.
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

# caches (prefer new SCRATCH setup, then legacy, then home fallback)
if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
elif [ -w "/shared/results/common/miksa/intact/SD" ] || [ -w "/shared/results/common/miksa/intact/SD/.cache" ]; then
    CACHE_BASE="/shared/results/common/miksa/intact/SD/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"

# Keep all tool caches consistent with selected CACHE_ROOT.
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
mkdir -p "$TMPDIR" "$WANDB_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/intact/pipeline_nsfw_debug.yaml}"
USE_WANDB="${USE_WANDB:-0}"

echo "Starting Flux NSFW debug run on $(hostname)"
echo "Using config: $CONFIG_PATH"
echo "USE_WANDB=$USE_WANDB"

echo "=== launching single debug run ==="
if [ "$USE_WANDB" = "1" ]; then
    python intact_pipeline.py --config "$CONFIG_PATH"
else
    python intact_pipeline.py --config "$CONFIG_PATH" --no-wandb
fi

echo "Flux NSFW debug run finished."
