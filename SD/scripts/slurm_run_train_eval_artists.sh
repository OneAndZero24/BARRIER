#!/bin/bash
# ============================================================================
# SLURM – Train Artists + LPIPS Evaluation (single run)
# ============================================================================
# Usage:
#   sbatch SD/scripts/slurm_run_train_eval_artists.sh
#   # or with custom config overrides:
#   sbatch SD/scripts/slurm_run_train_eval_artists.sh --epochs 5 --regular_scale 0.01
# ============================================================================

#SBATCH --job-name=sd-lpips-eval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxa100
#SBATCH --time=12:00:00

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
if [ -d "SD" ]; then
    cd SD
fi

# ---- Redirect caches to scratch / shared ----
if [ -n "$SCRATCH" ]; then
    export CACHE_ROOT="$SCRATCH/.cache"
else
    export CACHE_ROOT="/shared/results/common/miksa/intact/SD/.cache"
fi

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
export TMPDIR="$CACHE_ROOT/tmp"
export HUGGINGFACE_HUB_DISABLE_ENTRYPOINT_INTROSPECTION=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

export PYTHONPATH="${HOME}/InTAct-Unl:${PYTHONPATH:-}"

echo "================================================"
echo "  Artist LPIPS Eval – Job $SLURM_JOB_ID"
echo "  Directory: $(pwd)"
echo "  Args:     $@"
echo "================================================"

python scripts/lpips_eval_pipeline.py --config configs/pipeline_artists.yaml "$@"

echo "LPIPS Eval Pipeline Complete."
