#!/bin/bash -l
# ============================================================================
# SLURM – BARRIER Evaluation on ImageNet-Diversi50 (Table 3)
# ============================================================================
# Evaluates a trained BARRIER model on all 50 Diversi50 concepts + COCO CLIP.
# Matches ScaPre protocol EXACTLY:
#   - SD v1.5 pipeline (runwayml/stable-diffusion-v1-5)
#   - PNDM scheduler, 50 steps, cfg=7.5, 512x512
#   - ResNet-50 (ImageNet weights) with substring matching
#   - CLIP ViT-B/32 for CLIPcoco
#   - Same seeds from imagenet-50.csv
#
# Usage:
#   cd SD
#   sbatch scapre/scripts/slurm_eval_diversi50.sh
# ============================================================================

#SBATCH --job-name=eval-dv50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
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

# ---- Set your checkpoint path below ----
CKPT="$RESULTS_BASE/models/compvis-intact-rl_imagenet_diversi50-targets_*-lambda_*-epochs_*-lr_*/diffusers-*.pt"
CKPT=$(ls $CKPT 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $RESULTS_BASE/models/"
    echo "Run slurm_train_diversi50.sh first, or set CKPT manually."
    exit 1
fi

echo "============================================"
echo "  ScaPre Eval – Diversi50 (Table 3)"
echo "  Checkpoint: $CKPT"
echo "============================================"

python scapre/evaluate.py \
    --benchmark diversi50 \
    --ckpt_name "$CKPT" \
    --output_dir "$RESULTS_BASE/results" \
    --coco_prompts_source scapre/datasets/coco_30k.csv \
    --coco_max_images 5000

echo "Done.  Results: $RESULTS_BASE/results/diversi50/results_diversi50.json"
