#!/bin/bash -l
# ============================================================================
# SLURM Job – Vanilla Flux on Unsafe Prompts
# ============================================================================
#
# Generates images using vanilla Flux.1-dev (no unlearning, no LoRA) on
# specific unsafe prompts from the dataset:
# Case numbers: 296, 327, 649, 698, 1066, 1276, 1308
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_vanilla_flux_unsafe.sh
#
# Optional overrides at submit time:
#   sbatch --export=ALL,STEPS=50,OUTPUT_DIR=/path/to/output scripts/slurm_vanilla_flux_unsafe.sh
#
# ============================================================================

#SBATCH --job-name=flux-vanilla-unsafe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# Configuration parameters
STEPS="${STEPS:-28}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vanilla_flux}"
DEVICE="${DEVICE:-cuda:0}"
CSV_FILE="${CSV_FILE:-../SD/prompts/unsafe-prompts4703.csv}"

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
echo "Vanilla Flux – Unsafe Prompts Generation"
echo "============================================================"
echo "Hostname:        $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "GPU:             $DEVICE"
echo "Inference steps: $STEPS"
echo "Output dir:      $OUTPUT_DIR"
echo "CSV file:        $CSV_FILE"
echo "Cache root:      $CACHE_ROOT"
echo "============================================================"

# Run the Python script
python run_vanilla_flux_unsafe.py \
    --csv "$CSV_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --device "$DEVICE"

echo "============================================================"
echo "Job completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
