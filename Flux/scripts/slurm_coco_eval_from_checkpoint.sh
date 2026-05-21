#!/bin/bash -l
# ============================================================================
# SLURM Job – Flux COCO-only Evaluation from a Saved Checkpoint
# ============================================================================
# Loads a fine-tuned Flux checkpoint directly from a .safetensors path and
# runs only COCO FID + COCO CLIP evaluation.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_coco_eval_from_checkpoint.sh
#
# Optional overrides:
#   sbatch --export=MODEL_WEIGHTS_PATH=/path/to/model.safetensors scripts/slurm_coco_eval_from_checkpoint.sh
#   sbatch --export=CONFIG_PATH=configs/intact/pipeline_nsfw.yaml scripts/slurm_coco_eval_from_checkpoint.sh
#   sbatch --export=OUTPUT_DIR=/scratch/evals scripts/slurm_coco_eval_from_checkpoint.sh
# ============================================================================

#SBATCH --job-name=flux-coco-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Load Hugging Face credentials from the saved token file if present.
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

# caches (prefer new SCRATCH setup, then legacy, then home fallback)
if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
elif [ -w "/shared/results/common/miksa/intact/SD" ] || [ -w "/shared/results/common/miksa/intact/SD/.cache" ]; then
    CACHE_BASE="/shared/results/common/miksa/intact/SD/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"

# Keep all tool caches consistent with selected CACHE_ROOT.
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
mkdir -p "$TMPDIR" "$WANDB_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/intact/pipeline_nsfw.yaml}"
MODEL_WEIGHTS_PATH="${MODEL_WEIGHTS_PATH:-/net/scratch/hscra/plgrid/plgmiksa/models/flux-intact-nsfw-nude-targets_blk15-16-17-18_proj_proj-lambda_0.5-epochs_5-lr_5e-05.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-$CACHE_ROOT/flux-coco-eval}"
PREGENERATED_IMAGES_DIR="${PREGENERATED_IMAGES_DIR:-}"
Coco_ARGS=(
    --config "$CONFIG_PATH"
    --model-weights-path "$MODEL_WEIGHTS_PATH"
    --output-dir "$OUTPUT_DIR"
)

if ! python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("torchmetrics") else 1)
PY
then
    echo "torchmetrics not found, installing into the active environment"
    python -m pip install --upgrade torchmetrics
fi

if [ -n "${COCO_PROMPTS_CSV:-}" ]; then
    Coco_ARGS+=(--coco-prompts-csv "$COCO_PROMPTS_CSV")
fi
if [ -n "${COCO_ANN_PATH:-}" ]; then
    Coco_ARGS+=(--coco-ann-path "$COCO_ANN_PATH")
fi
if [ -n "${COCO_IMAGES_DIR:-}" ]; then
    Coco_ARGS+=(--coco-images-dir "$COCO_IMAGES_DIR")
fi
if [ -n "${PREGENERATED_IMAGES_DIR:-}" ]; then
    Coco_ARGS+=(--pregenerated-images-dir "$PREGENERATED_IMAGES_DIR")
fi
if [ -n "${DEVICE:-}" ]; then
    Coco_ARGS+=(--device "$DEVICE")
fi

if [ -n "${N_CAPTIONS:-}" ]; then
    Coco_ARGS+=(--n-captions "$N_CAPTIONS")
fi
if [ -n "${NUM_SAMPLES_PER_PROMPT:-}" ]; then
    Coco_ARGS+=(--num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT")
fi
if [ -n "${GENERATION_BATCH_SIZE:-}" ]; then
    Coco_ARGS+=(--generation-batch-size "$GENERATION_BATCH_SIZE")
fi
if [ -n "${GUIDANCE_SCALE:-}" ]; then
    Coco_ARGS+=(--guidance-scale "$GUIDANCE_SCALE")
fi
if [ -n "${DDIM_STEPS:-}" ]; then
    Coco_ARGS+=(--ddim-steps "$DDIM_STEPS")
fi
if [ -n "${IMAGE_SIZE:-}" ]; then
    Coco_ARGS+=(--image-size "$IMAGE_SIZE")
fi
if [ -n "${MAX_PROMPTS:-}" ]; then
    Coco_ARGS+=(--max-prompts "$MAX_PROMPTS")
fi
if [ -n "${FID_FEATURE:-}" ]; then
    Coco_ARGS+=(--fid-feature "$FID_FEATURE")
fi
if [ -n "${MAX_REAL:-}" ]; then
    Coco_ARGS+=(--max-real "$MAX_REAL")
fi
if [ -n "${MAX_FAKE:-}" ]; then
    Coco_ARGS+=(--max-fake "$MAX_FAKE")
fi

echo "Starting Flux COCO-only evaluation on $(hostname)"
echo "Checkpoint: $MODEL_WEIGHTS_PATH"
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"

python scripts/run_coco_eval_from_checkpoint.py "${Coco_ARGS[@]}"