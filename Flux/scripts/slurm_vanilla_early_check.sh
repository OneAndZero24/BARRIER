#!/bin/bash -l
# ============================================================================
# SLURM Job – EraseAnything Early NSFW Check
# ============================================================================
#
# Generates early-check prompts (nude vs clothed) using EraseAnything LoRA weights
# (Flux-erase-dev/pytorch_lora_weights.safetensors), with sampling settings from config.
# Compares unlearned model against base Flux baseline.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_vanilla_early_check.sh
#
# Optional overrides:
#   sbatch --export=ALL,CONFIG_PATH=configs/intact/pipeline_nsfw.yaml,STEPS=50,GUIDANCE_SCALE=7.5 scripts/slurm_vanilla_early_check.sh
# ============================================================================

#SBATCH --job-name=flux-vanilla-early
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# Configuration parameters
CONFIG_PATH="${CONFIG_PATH:-configs/intact/pipeline_nsfw.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/esd_early_nsfw_check}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

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

# Caches (prefer SCRATCH, then home fallback)
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
echo "Vanilla Flux – Early NSFW Check (EraseAnything LoRA)"
echo "============================================================"
echo "Hostname:        $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "GPU:             $DEVICE"
echo "Model:           Flux-erase-dev/pytorch_lora_weights.safetensors"
echo "Config:          $CONFIG_PATH"
echo "Output dir:      $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then echo "Samples/prompt:  $NUM_SAMPLES (override)"; else echo "Samples/prompt:  from config"; fi
if [ -n "$STEPS" ]; then echo "Steps:           $STEPS (override)"; else echo "Steps:           from config"; fi
if [ -n "$GUIDANCE_SCALE" ]; then echo "Guidance scale:  $GUIDANCE_SCALE (override)"; else echo "Guidance scale:  from config"; fi
if [ -n "$IMAGE_SIZE" ]; then echo "Image size:      $IMAGE_SIZE (override)"; else echo "Image size:      from config"; fi
echo "Cache root:      $CACHE_ROOT"
echo "============================================================"

CMD=(python scripts/generate_vanilla_early_check.py
    --config "$CONFIG_PATH"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE")

if [ -n "$STEPS" ]; then CMD+=(--steps "$STEPS"); fi
if [ -n "$GUIDANCE_SCALE" ]; then CMD+=(--guidance-scale "$GUIDANCE_SCALE"); fi
if [ -n "$IMAGE_SIZE" ]; then CMD+=(--image-size "$IMAGE_SIZE"); fi
if [ -n "$NUM_SAMPLES" ]; then CMD+=(--num-samples "$NUM_SAMPLES"); fi

"${CMD[@]}"

echo "============================================================"
echo "Job completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"