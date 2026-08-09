#!/bin/bash -l
# ============================================================================
# SLURM – BARRIER SVD Separation Scatter (best 2 dims per layer by IoU)
# ============================================================================
# For each protected layer, finds the TWO SVD dimensions with the LOWEST IoU
# between forget (NSFW) and remain (SFW) activations, then produces 2D scatter
# plots with zone occupancy breakdown.
#
# Metrics reported per layer:
#   - Best dim pair + 2D IoU
#   - Zone occupancy (inside_box / neg_inf / pos_inf / outside) for forget & remain
#
# Outputs: svd_separation.json, svd_separation_<layer>.png per layer,
#          iou_rankings.png
#
# HOW TO USE:
#   cd SD
#   sbatch scripts/slurm_svd_separation_scatter.sh
# ============================================================================

#SBATCH --job-name=sd-svd-sep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --qos=quick

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

# Target layers to analyze (attn2 cross-attention)
TARGETS="attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0"

# SVD hyperparameters
REDUCED_DIM=32
LOWER_PERCENTILE=0.05
UPPER_PERCENTILE=0.95

# Number of batches for SVD calibration and projection collection
SVD_BATCHES=50
FORGET_BATCHES=50
REMAIN_BATCHES=50

# IoU search parameters
TOP_K_1D=10
N_BINS_2D=50

USE_ACTUAL_BOUNDS=1  # 1 = true, 0 = false (include remain data in bounds)

NSFW_DATA_PATH="/shared/results/common/miksa/intact/SD/data/nsfw"
NOT_NSFW_DATA_PATH="/shared/results/common/miksa/intact/SD/data/not-nsfw"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="/shared/results/common/miksa/intact/SD/svd_separation"
OUT_DIR="${RESULTS_BASE}/${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# ============================================================================
# Run analysis
# ============================================================================
echo "============================================"
echo "BARRIER SVD Separation Scatter Analysis"
echo "  Host:       $(hostname)"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "============================================"
echo "  Targets:     ${TARGETS}"
echo "  Reduced dim: ${REDUCED_DIM}"
echo "  Percentiles: ${LOWER_PERCENTILE} / ${UPPER_PERCENTILE}"
echo "  Batches:     svd=${SVD_BATCHES}  forget=${FORGET_BATCHES}  remain=${REMAIN_BATCHES}"
echo "  IoU search:  top_k_1d=${TOP_K_1D}  n_bins_2d=${N_BINS_2D}"
echo "  Actual bnds: ${USE_ACTUAL_BOUNDS}"
echo "  CKPT:        ${CKPT_PATH}"
echo "  Out dir:     ${OUT_DIR}"
echo "============================================"
echo ""

python scripts/svd_separation_scatter.py \
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
    --top_k_1d "${TOP_K_1D}" \
    --n_bins_2d "${N_BINS_2D}" \
    $([ "${USE_ACTUAL_BOUNDS}" = "1" ] && echo "--use_actual_bounds" || echo "--no_actual_bounds") \
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
