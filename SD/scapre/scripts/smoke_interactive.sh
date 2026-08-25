#!/bin/bash -l
# ============================================================================
# BARRIER ScaPre grid SMOKE TEST (both benchmarks: Confuse5 + Diversi50)
# ============================================================================
# Validates the full grid pipeline (train -> InTAct -> save -> evaluate) with
# tiny capped settings so it finishes in ~20-40 min on one GPU and can be run
# inside an interactive session (or via sbatch with a short walltime).
#
# Usage:
#   cd "$HOME/InTAct-Unl/SD"
#   bash scapre/scripts/smoke_interactive.sh
#
# Tuning (env overrides):
#   SMOKE_STEPS=15        optimizer steps per benchmark
#   SMOKE_PNG=2           generated images per concept in eval
#   SMOKE_COCO=100        COCO images for CLIPcoco eval
#   SMOKE_MODELS_DIR=...  where checkpoints land (default under $SCRATCH)
# ============================================================================

set -uo pipefail

# ---- Environment (same as the real grid scripts) ----
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
export WANDB_DATA_DIR="$CACHE_ROOT/wandb_data"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_DATA_DIR"

# ---- Paths (same as grid scripts) ----
IMAGENET_ROOT="$SCRATCH/data/ImageNet"
CKPT="$SCRATCH/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
CONFIG=configs/stable-diffusion/v1-intact.yaml
DIFFUSERS_CFG=diffusers_unet_config.json
COCO_CSV=scapre/datasets/coco_30k.csv

SMOKE_BASE="${SMOKE_MODELS_DIR:-$SCRATCH/intact/SD/scapre/smoke}"
SMOKE_MODELS="$SMOKE_BASE/models"
SMOKE_RESULTS="$SMOKE_BASE/results"
mkdir -p "$SMOKE_MODELS" "$SMOKE_RESULTS"

SMOKE_STEPS="${SMOKE_STEPS:-15}"
SMOKE_PNG="${SMOKE_PNG:-2}"
SMOKE_COCO="${SMOKE_COCO:-100}"

# Same explicit attention targets as the grid scripts (blocks 6 & 8).
TARGETS=(output_blocks.6.1.transformer_blocks.0.attn2.to_q \
         output_blocks.6.1.transformer_blocks.0.attn2.to_k \
         output_blocks.6.1.transformer_blocks.0.attn2.to_v \
         output_blocks.8.1.transformer_blocks.0.attn2.to_q \
         output_blocks.8.1.transformer_blocks.0.attn2.to_k \
         output_blocks.8.1.transformer_blocks.0.attn2.to_v)

FAILURES=()

run_stage() {
    local stage="$1"; shift
    echo "==================================================================="
    echo "[SMOKE] $stage"
    echo "==================================================================="
    if "$@"; then
        echo "[SMOKE-PASS] $stage"
    else
        echo "[SMOKE-FAIL] $stage"
        FAILURES+=("$stage")
    fi
}

# 1) Preflight -- imports (bug: imagenet_data not on sys.path) + concept mapping
run_stage "preflight imports" python -c \
    "import torch; from ldm.util import instantiate_from_config; import scapre.train, scapre.evaluate; print('imports ok')"

run_stage "preflight ImageNet concepts" python -c \
    "from scapre.imagenet_data import DIVERSI50_CONCEPTS, CONFUSE5_CONCEPTS, build_imagenet_class_index; n2i,_=build_imagenet_class_index('$IMAGENET_ROOT'); print('missing:', [c for c in DIVERSI50_CONCEPTS+CONFUSE5_CONCEPTS if c.lower() not in n2i])"

TRAIN_ARGS=(--imagenet_root "$IMAGENET_ROOT" \
            --ckpt_path "$CKPT" \
            --config_path "$CONFIG" \
            --diffusers_config_path "$DIFFUSERS_CFG" \
            --base_method rl \
            --lr 5e-6 \
            --batch_size 8 \
            --targets "${TARGETS[@]}" \
            --lambda_interval 5.0 \
            --reduced_dim 32 \
            --infinity_scale 18.0 \
            --use_actual_bounds \
            --epochs 1 \
            --max_steps "$SMOKE_STEPS" \
            --model_save_dir "$SMOKE_MODELS")

# 2) Confuse5: grid uses bounds_fraction 0.5 / remain 0.1 -- smoke shrinks both.
C5_NAME=smoke-c5
run_stage "Confuse5 train ($SMOKE_STEPS steps)" python scapre/train.py \
    --benchmark confuse5 "${TRAIN_ARGS[@]}" \
    --bounds_fraction 0.01 --bounds_remain_fraction 0.005 \
    --model_name "$C5_NAME"

sleep 5   # let Lustre/NFS flush the checkpoint listing
C5_CKPT=$(ls "$SMOKE_MODELS/$C5_NAME"/diffusers-*.pt 2>/dev/null | head -1)
if [ -n "$C5_CKPT" ]; then
    run_stage "Confuse5 eval (${SMOKE_PNG}/concept)" python scapre/evaluate.py \
        --benchmark confuse5 \
        --ckpt_name "$C5_CKPT" \
        --output_dir "$SMOKE_RESULTS/$C5_NAME" \
        --max_prompts_per_concept "$SMOKE_PNG" \
        --coco_prompts_source "$COCO_CSV" \
        --coco_max_images "$SMOKE_COCO"
else
    echo "[SMOKE-FAIL] Confuse5 eval (no checkpoint produced)"
    FAILURES+=("Confuse5 eval")
fi

# 3) Diversi50: grid uses bounds_fraction 0.1 / remain 0.05 -- smoke shrinks both.
DV_NAME=smoke-dv50
run_stage "Diversi50 train ($SMOKE_STEPS steps)" python scapre/train.py \
    --benchmark diversi50 "${TRAIN_ARGS[@]}" \
    --bounds_fraction 0.01 --bounds_remain_fraction 0.005 \
    --model_name "$DV_NAME"

sleep 5   # let Lustre/NFS flush the checkpoint listing
DV_CKPT=$(ls "$SMOKE_MODELS/$DV_NAME"/diffusers-*.pt 2>/dev/null | head -1)
if [ -n "$DV_CKPT" ]; then
    run_stage "Diversi50 eval (${SMOKE_PNG}/concept)" python scapre/evaluate.py \
        --benchmark diversi50 \
        --ckpt_name "$DV_CKPT" \
        --output_dir "$SMOKE_RESULTS/$DV_NAME" \
        --max_prompts_per_concept "$SMOKE_PNG" \
        --coco_prompts_source "$COCO_CSV" \
        --coco_max_images "$SMOKE_COCO"
else
    echo "[SMOKE-FAIL] Diversi50 eval (no checkpoint produced)"
    FAILURES+=("Diversi50 eval")
fi

# ---- Summary ----
echo "==================================================================="
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "[SMOKE] ALL STAGES PASSED -- both grids are structurally OK to launch."
    echo "[SMOKE] Models in: $SMOKE_MODELS"
    exit 0
else
    echo "[SMOKE] FAILED STAGES: ${FAILURES[*]}"
    exit 1
fi
