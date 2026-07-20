#!/bin/bash
# ============================================================================
# SLURM Array Job – BARRIER Grid Search on ImageNet-Confuse5
# ============================================================================
# Grid over 24 hyperparameter combos (4 lambdas x 2 epochs x 3 LRs).
# Each array job: train BARRIER on all 10 Confuse5 concepts,
# then evaluate using ScaPre protocol (Table 4).
#
# Usage:
#   sbatch SD/scapre/scripts/slurm_grid_confuse5.sh
# ============================================================================

#SBATCH --job-name=barrier-grid-c5
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=dgxh100
#SBATCH --array=0-23

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
export RESULTS_ROOT=/shared/results/common/miksa/intact/SD
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT"

# ============================================================================
# Grid dimensions: 4 lambdas  x  2 epochs  x  3 LRs  =  24 combos
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
LR_IDX=$(( TMP_IDX % NUM_LRS ));  TMP_IDX=$(( TMP_IDX / NUM_LRS ))
EPOCH_IDX=$(( TMP_IDX % NUM_EPOCHS )); TMP_IDX=$(( TMP_IDX / NUM_EPOCHS ))
LAMBDA_IDX=${TMP_IDX}

LAMBDA=${GRID_LAMBDAS[$LAMBDA_IDX]}
EPOCH=${GRID_EPOCHS[$EPOCH_IDX]}
LR=${GRID_LRS[$LR_IDX]}

echo "============================================"
echo "Confuse5 Grid – combo ${TASK_ID}"
echo "  lambda=${LAMBDA}  epochs=${EPOCH}  lr=${LR}"
echo "============================================"

# ---- Train ----
MODEL_NAME="barrier-c5-lam${LAMBDA}-ep${EPOCH}-lr${LR}"

python scapre/train.py \
    --benchmark confuse5 \
    --imagenet_root /datasets/ImageNet \
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
    --model_save_dir "${RESULTS_ROOT}/models/scapre-c5-grid"

CKPT=$(find "${RESULTS_ROOT}/models/scapre-c5-grid/${MODEL_NAME}" -name "diffusers-*.pt" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No diffusers checkpoint found"
    exit 1
fi
echo "Checkpoint: $CKPT"

# ---- Evaluate (Table 4) ----
python scapre/evaluate.py \
    --benchmark confuse5 \
    --ckpt_name "$CKPT" \
    --output_dir "${RESULTS_ROOT}/results/scapre-c5-grid/${MODEL_NAME}" \
    --coco_prompts_source scapre/datasets/coco_30k.csv \
    --coco_max_images 5000 \
    --max_prompts_per_concept 100

echo "Confuse5 grid combo ${TASK_ID} complete."
