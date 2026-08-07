#!/bin/bash -l
# ============================================================================
# SLURM – BARRIER Activation Space Occupancy Analysis (Helios / PLGrid)
# ============================================================================
# Runs on the BASE (pre-unlearning) SD model to check whether forget and
# remain activations occupy distinct regions in the SVD subspace.
#
# Zone classification (per SVD dimension, bounds estimated from forget data):
#   below_range   : x < inf_low
#   negative_space: inf_low ≤ x < z_min          ← "safe zone" for remain
#   inside_box    : z_min  ≤ x ≤ z_max           ← the forget box
#   positive_space: z_max  < x ≤ inf_high        ← "safe zone" for remain
#   above_range   : x > inf_high
#
# Metrics reported: per-token mean fraction of dims in each zone.
#
# Outputs: activation_analysis.json, density_histograms.png,
#          zone_breakdown.png, mean_overlap_bars.png,
#          svd_scatter.png, per_dim_remain_zones.png
#
# HOW TO USE:
#   cd SD
#   sbatch scripts/slurm_activation_space_analysis.sh
# ============================================================================

#SBATCH --job-name=sd-act-space
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxh100
#SBATCH --qos=big

set -euo pipefail

# ---- Environment ----
source /home/miksa/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl/taming-transformers:$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Hugging Face token
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/shared/results/common/miksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

CACHE_BASE="/shared/results/common/miksa/.cache"
export CACHE_ROOT="$CACHE_BASE"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

# ============================================================================
# Analysis parameters
# ============================================================================
DEVICE=0
CONFIG_PATH="configs/stable-diffusion/v1-intact.yaml"
CKPT_PATH="$HOME/InTAct-Unl/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
IMAGE_SIZE=512
BATCH_SIZE=4

# Target layers to analyze (attn2 cross-attention — same as typical intact run)
TARGETS="attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0"

# SVD hyperparameters (must match typical InTAct config)
REDUCED_DIM=32
LOWER_PERCENTILE=0.05
UPPER_PERCENTILE=0.95

# Number of batches for SVD calibration and projection collection
SVD_BATCHES=50
FORGET_BATCHES=50
REMAIN_BATCHES=50

NSFW_DATA_PATH="/shared/results/common/miksa/intact/SD/data/nsfw"
NOT_NSFW_DATA_PATH="/shared/results/common/miksa/intact/SD/data/not-nsfw"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="/shared/results/common/miksa/intact/SD/activation_space"
OUT_DIR="${RESULTS_BASE}/${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# ============================================================================
# Run analysis
# ============================================================================
echo "============================================"
echo "BARRIER Activation Space Occupancy Analysis"
echo "  Host:       $(hostname)"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "============================================"
echo "  Targets:     ${TARGETS}"
echo "  Reduced dim: ${REDUCED_DIM}"
echo "  Percentiles: ${LOWER_PERCENTILE} / ${UPPER_PERCENTILE}"
echo "  Batches:     svd=${SVD_BATCHES}  forget=${FORGET_BATCHES}  remain=${REMAIN_BATCHES}"
echo "  CKPT:        ${CKPT_PATH}"
echo "  Out dir:     ${OUT_DIR}"
echo "============================================"
echo ""

python scripts/activation_space_analysis.py \
    --device "${DEVICE}" \
    --config_path "${CONFIG_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --targets ${TARGETS} \
    --reduced_dim "${REDUCED_DIM}" \
    --lower_percentile "${LOWER_PERCENTILE}" \
    --upper_percentile "${UPPER_PERCENTILE}" \
    --svd_batches "${SVD_BATCHES}" \
    --forget_batches "${FORGET_BATCHES}" \
    --remain_batches "${REMAIN_BATCHES}" \
    --nsfw_data_path "${NSFW_DATA_PATH}" \
    --not_nsfw_data_path "${NOT_NSFW_DATA_PATH}" \
    --out_dir "${OUT_DIR}"

ret=$?

echo ""
echo "============================================"
if [ $ret -eq 0 ]; then
    echo "Analysis complete. Results:"
    ls -la "${OUT_DIR}/"
else
    echo "Analysis FAILED with exit code ${ret}"
fi
echo "============================================"

exit $ret
