#!/bin/bash -l
# ============================================================================
# SLURM Job – Vanilla Flux I2P (full unsafe prompts) 
# ============================================================================
# Generates I2P (EraseAnything) evaluation images using the vanilla Flux base
# model (no unlearning), with the same sampling settings as the InTAct pipeline.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_vanilla_flux_i2p.sh
#
# Optional overrides:
#   sbatch --export=ALL,CONFIG_PATH=configs/intact/pipeline_nsfw.yaml,OUTPUT_DIR=/tmp/vanilla_i2p scripts/slurm_vanilla_flux_i2p.sh
# ============================================================================

#SBATCH --job-name=flux-vanilla-i2p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/intact/pipeline_nsfw.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vanilla_i2p}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
PROMPTS_CSV="${PROMPTS_CSV:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

# ---- Environment ----
ml ML-bundle/25.10
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Load Hugging Face credentials
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
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR"

echo "============================================================"
echo "Vanilla Flux – I2P generation"
echo "============================================================"
echo "Hostname:        $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Config:          $CONFIG_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "Device:          $DEVICE"
echo "Steps (override): $STEPS"
echo "Guidance (override): $GUIDANCE_SCALE"
echo "Image size (override): $IMAGE_SIZE"
echo "Num samples (override): $NUM_SAMPLES"
echo "Batch size (override): $BATCH_SIZE"
echo "Prompts CSV (override): $PROMPTS_CSV"
echo "Base model (override): $BASE_MODEL_PATH"
echo "Cache root:      $CACHE_ROOT"
echo "============================================================"

# Build command with optional overrides (script reads config if not provided)
CMD=(python scripts/generate_vanilla_i2p.py --config "$CONFIG_PATH")
if [ -n "$PROMPTS_CSV" ]; then CMD+=(--prompts "$PROMPTS_CSV"); fi
if [ -n "$OUTPUT_DIR" ]; then CMD+=(--output-dir "$OUTPUT_DIR"); fi
if [ -n "$DEVICE" ]; then CMD+=(--device "$DEVICE"); fi
if [ -n "$STEPS" ]; then CMD+=(--steps "$STEPS"); fi
if [ -n "$GUIDANCE_SCALE" ]; then CMD+=(--guidance-scale "$GUIDANCE_SCALE"); fi
if [ -n "$IMAGE_SIZE" ]; then CMD+=(--image-size "$IMAGE_SIZE"); fi
if [ -n "$NUM_SAMPLES" ]; then CMD+=(--num-samples "$NUM_SAMPLES"); fi
if [ -n "$BATCH_SIZE" ]; then CMD+=(--batch-size "$BATCH_SIZE"); fi
if [ -n "$BASE_MODEL_PATH" ]; then CMD+=(--base-model-path "$BASE_MODEL_PATH"); fi

"${CMD[@]}"

echo "============================================================"
echo "Job completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
