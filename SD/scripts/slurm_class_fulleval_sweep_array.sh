#!/bin/bash
# ============================================================================
# SLURM array launcher for the SD class-forgetting W&B sweep
# ============================================================================
# Usage:
#   cd SD
#   sbatch --array=0-3 scripts/slurm_class_fulleval_sweep_array.sh gmpa2plj
#
# Each array task starts one W&B agent and asks it to consume a single run.
# Re-submit the same sweep id if the job times out; completed class metrics are
# reused from disk, so the same trial can continue where it stopped.
# ============================================================================

#SBATCH --job-name=sd-class-sweep-array
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100

set -euo pipefail

WANDB_ENTITY="oneandzero24"
WANDB_PROJECT="intact-sd"
RUNS_PER_AGENT="${2:-1}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <wandb-sweep-id> [runs-per-agent]"
  exit 1
fi

SWEEP_ID="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
# Disables cache-locking that breaks on concurrent NFS mounts
export HUGGINGFACE_HUB_DISABLE_ENTRYPOINT_INTROSPECTION=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Starting W&B agent for sweep ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
wandb agent --count "$RUNS_PER_AGENT" "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"