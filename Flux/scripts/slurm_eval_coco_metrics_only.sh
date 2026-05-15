#!/bin/bash -l
# ============================================================================
# SLURM Job – Compute COCO FID + CLIP from pregenerated images
# ============================================================================
# Lightweight metrics-only evaluation. Requires images and prompts CSV.
#
# Usage:
#   cd Flux
#   sbatch --export=IMAGES_DIR=/path/to/images,PROMPTS_CSV=/path/to/prompts.csv scripts/slurm_eval_coco_metrics_only.sh
#
# Environment variables:
#   IMAGES_DIR (required)       – directory with pregenerated images
#   PROMPTS_CSV (required)      – CSV with case_number and prompt columns
#   COCO_IMAGES_DIR (optional)  – real COCO val images for FID reference
#   COCO_ANN_PATH (optional)    – COCO captions JSON for FID reference
#   DEVICE (optional)           – CUDA device (default: cuda:0)
#   IMAGE_SIZE (optional)       – image size for FID (default: 512)
#   FID_FEATURE (optional)      – InceptionV3 feature dimension (default: 2048)
#   MAX_REAL (optional)         – max real images for FID
#   MAX_FAKE (optional)         – max fake images for FID
# ============================================================================

#SBATCH --job-name=flux-metrics-only
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# Require IMAGES_DIR and PROMPTS_CSV
if [ -z "${IMAGES_DIR:-}" ]; then
    echo "ERROR: IMAGES_DIR not set"
    exit 1
fi

if [ -z "${PROMPTS_CSV:-}" ]; then
    echo "ERROR: PROMPTS_CSV not set"
    exit 1
fi

# ---- Environment ----
ml ML-bundle/24.06a
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Pre-install torch-fidelity and torchmetrics for FID computation
echo "Ensuring torch-fidelity and torchmetrics are installed..."
python -m pip install --quiet --upgrade "torch-fidelity>=0.3.0" "torchmetrics[image]>=1.0" 2>&1 | grep -v "WARNING:" | grep -v "already satisfied" || true

# Caches
if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
mkdir -p "$CACHE_ROOT/tmp" "$WANDB_DIR"

# Load Hugging Face token if available
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi

DEVICE="${DEVICE:-cuda:0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
FID_FEATURE="${FID_FEATURE:-2048}"

ARGS=(
    --images-dir "$IMAGES_DIR"
    --prompts-csv "$PROMPTS_CSV"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --fid-feature "$FID_FEATURE"
)

if [ -n "${COCO_IMAGES_DIR:-}" ]; then
    ARGS+=(--coco-images-dir "$COCO_IMAGES_DIR")
fi
if [ -n "${COCO_ANN_PATH:-}" ]; then
    ARGS+=(--coco-ann-path "$COCO_ANN_PATH")
fi
if [ -n "${MAX_REAL:-}" ]; then
    ARGS+=(--max-real "$MAX_REAL")
fi
if [ -n "${MAX_FAKE:-}" ]; then
    ARGS+=(--max-fake "$MAX_FAKE")
fi

echo "Starting metrics-only COCO evaluation on $(hostname)"
echo "Images:    $IMAGES_DIR"
echo "Prompts:   $PROMPTS_CSV"
echo "Device:    $DEVICE"
echo ""

python scripts/eval_coco_metrics_only.py "${ARGS[@]}"
