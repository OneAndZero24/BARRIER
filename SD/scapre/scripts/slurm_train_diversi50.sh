#!/bin/bash -l
# ============================================================================
# SLURM – BARRIER Training on ImageNet-Diversi50 (50 concepts)
# ============================================================================
# Trains BARRIER/InTAct on all 50 Diversi50 concepts using ImageNet-1K images.
# Produces a diffusers UNet checkpoint for ScaPre-compatible evaluation.
#
# Usage:
#   cd SD
#   sbatch scapre/scripts/slurm_train_diversi50.sh
# ============================================================================

#SBATCH --job-name=barrier-dv50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
source "$SCRATCH/sd_venv/bin/activate"
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl/taming-transformers:$HOME/InTAct-Unl:${PYTHONPATH:-}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

RESULTS_BASE="${SCRATCH:-$HOME}/intact/SD/scapre"
mkdir -p "$RESULTS_BASE"

echo "============================================"
echo "BARRIER Diversi50 Training on $(hostname)"
echo "============================================"

# ---- Train ----
python scapre/train.py \
    --benchmark diversi50 \
    --imagenet_root "$SCRATCH/data/ImageNet" \
    --ckpt_path "$SCRATCH/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt" \
    --config_path configs/stable-diffusion/v1-intact.yaml \
    --diffusers_config_path diffusers_unet_config.json \
    --base_method rl \
    --lr 5e-6 \
    --epochs 5 \
    --batch_size 8 \
    --targets to_q to_k to_v \
    --lambda_interval 4.0 \
    --reduced_dim 32 \
    --infinity_scale 18.0 \
    --use_actual_bounds \
    --bounds_fraction 0.3 \
    --model_save_dir "$RESULTS_BASE/models"

echo "Training complete.  Evaluate with:"
echo "  python scapre/evaluate.py --benchmark diversi50 --ckpt_name <diffusers-pt-path>"
