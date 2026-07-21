#!/bin/bash -l
# ============================================================================
# SLURM Array Job – BARRIER Grid Search on ImageNet-Confuse5
# ============================================================================
# Grid over 24 hyperparameter combos: 4 lambdas x 2 epochs x 3 LRs.
# Each job: train → evaluate using ScaPre protocol (Table 4).
#
# Usage:
#   cd SD
#   sbatch scapre/scripts/slurm_grid_confuse5.sh
# ============================================================================

#SBATCH --job-name=bar-g-c5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-23

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

# ============================================================================
# Grid: 4 lambdas x 2 epochs x 3 LRs = 24 combos
# ============================================================================
GRID_LAMBDAS=(0.5 1.0 5.0 10.0)
GRID_EPOCHS=(3 5)
GRID_LRS=(1e-6 5e-6 1e-5)

NUM_LAMBDAS=${#GRID_LAMBDAS[@]}
NUM_EPOCHS=${#GRID_EPOCHS[@]}
NUM_LRS=${#GRID_LRS[@]}
NUM_COMBOS=$(( NUM_LAMBDAS * NUM_EPOCHS * NUM_LRS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID >= NUM_COMBOS )); then
    echo "Task ${TASK_ID} out of range [0, $((NUM_COMBOS - 1))]. Exiting."
    exit 0
fi

TMP_IDX=${TASK_ID}
LR_IDX=$(( TMP_IDX % NUM_LRS ));    TMP_IDX=$(( TMP_IDX / NUM_LRS ))
EPOCH_IDX=$(( TMP_IDX % NUM_EPOCHS )); TMP_IDX=$(( TMP_IDX / NUM_EPOCHS ))
LAMBDA_IDX=${TMP_IDX}

LAMBDA=${GRID_LAMBDAS[$LAMBDA_IDX]}
EPOCH=${GRID_EPOCHS[$EPOCH_IDX]}
LR=${GRID_LRS[$LR_IDX]}

echo "============================================"
echo "Confuse5 Grid – combo ${TASK_ID} on $(hostname)"
echo "  lambda=${LAMBDA}  epochs=${EPOCH}  lr=${LR}"
echo "============================================"

# ---- Train ----
MODEL_NAME="barrier-c5-lam${LAMBDA}-ep${EPOCH}-lr${LR}"

python scapre/train.py \
    --benchmark confuse5 \
    --imagenet_root "$SCRATCH/data/ImageNet" \
    --ckpt_path "$SCRATCH/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt" \
    --config_path configs/stable-diffusion/v1-intact.yaml \
    --diffusers_config_path diffusers_unet_config.json \
    --base_method rl \
    --lr "$LR" \
    --epochs "$EPOCH" \
    --batch_size 8 \
    --targets to_q to_k to_v \
    --lambda_interval "$LAMBDA" \
    --reduced_dim 32 \
    --infinity_scale 18.0 \
    --use_actual_bounds \
    --bounds_fraction 0.5 \
    --model_save_dir "$RESULTS_BASE/grid-models"

CKPT=$(ls "$RESULTS_BASE/grid-models/$MODEL_NAME"/diffusers-*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found for $MODEL_NAME"
    exit 1
fi
echo "Checkpoint: $CKPT"

# ---- Evaluate (Table 4) ----
python scapre/evaluate.py \
    --benchmark confuse5 \
    --ckpt_name "$CKPT" \
    --output_dir "$RESULTS_BASE/grid-results/$MODEL_NAME" \
    --coco_prompts_source scapre/datasets/coco_30k.csv \
    --coco_max_images 5000 \
    --max_prompts_per_concept 100

echo "Confuse5 grid combo ${TASK_ID} (${MODEL_NAME}) done."
