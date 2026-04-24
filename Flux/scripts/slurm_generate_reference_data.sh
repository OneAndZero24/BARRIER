#!/bin/bash -l
# ============================================================================
# SLURM Job – Generate Flux Reference Data (NSFW / Not-NSFW)
# ============================================================================
# 
# Generates synthetic reference datasets using Flux base model for InTAct
# activation bounds computation. Replaces old SD-generated data.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_generate_reference_data.sh
#
# Optional overrides at submit time:
#   sbatch --export=ALL,NUM_SAMPLES=800,BATCH_SIZE=2 scripts/slurm_generate_reference_data.sh
#
# This job runs on a single GPU and generates 1600 images total
# (800 NSFW + 800 NOT-NSFW, matching SD setup).
# ============================================================================

#SBATCH --job-name=flux-gen-ref-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-a40

set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-800}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
DDIM_STEPS="${DDIM_STEPS:-28}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

# ---- Environment ----
ml ML-bundle/24.06a
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

echo "Starting Flux reference data generation..."

python -c "import setup_cache; print('Cache configured')"

if [ -z "${SCRATCH:-}" ]; then
    echo "ERROR: SCRATCH is not set"
    exit 1
fi

mkdir -p "$SCRATCH/data/nsfw" "$SCRATCH/data/not-nsfw"

tmp_nsfw_csv="$(mktemp /tmp/flux_nsfw_prompts.XXXXXX.csv)"
tmp_not_nsfw_csv="$(mktemp /tmp/flux_not_nsfw_prompts.XXXXXX.csv)"
cleanup() {
    rm -f "$tmp_nsfw_csv" "$tmp_not_nsfw_csv"
}
trap cleanup EXIT

cat > "$tmp_nsfw_csv" << 'EOF'
case_number,prompt,evaluation_seed
1,a photo of a nude person,42
EOF

cat > "$tmp_not_nsfw_csv" << 'EOF'
case_number,prompt,evaluation_seed
1,a photo of a person wearing clothes,42
EOF

echo "Generating NSFW reference images: ${NUM_SAMPLES}"
python eval/generate_images.py \
    --model_name "" \
    --prompts_path "$tmp_nsfw_csv" \
    --save_path "$SCRATCH/data/nsfw" \
    --batch_size "$BATCH_SIZE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --ddim_steps "$DDIM_STEPS" \
    --image_size "$IMAGE_SIZE" \
    --num_samples "$NUM_SAMPLES"

echo "Generating NOT-NSFW reference images: ${NUM_SAMPLES}"
python eval/generate_images.py \
    --model_name "" \
    --prompts_path "$tmp_not_nsfw_csv" \
    --save_path "$SCRATCH/data/not-nsfw" \
    --batch_size "$BATCH_SIZE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --ddim_steps "$DDIM_STEPS" \
    --image_size "$IMAGE_SIZE" \
    --num_samples "$NUM_SAMPLES"

echo "Done!"
