#!/bin/bash
# ============================================================================
# SLURM Array – Artist Unlearning + LPIPS Sweep Agents
# ============================================================================
# Two-step usage:
#   Step 1 (on login node):
#     cd SD && wandb sweep --project intact-sd --entity oneandzero24 configs/sweep_artists_lpips.yaml
#     # Note the sweep ID from output, e.g. "gMpA2plj"
#
#   Step 2:
#     sbatch --array=0-19 SD/scripts/slurm_artist_sweep_array.sh <SWEEP_ID> [runs_per_agent]
# ============================================================================

#SBATCH --job-name=sd-lpips-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-30

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

# ---- Cache redirects ----
export CACHE_ROOT="/shared/results/common/miksa/intact/SD/.cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
export HUGGINGFACE_HUB_DISABLE_ENTRYPOINT_INTROSPECTION=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

echo "Array task: ${SLURM_ARRAY_TASK_ID:-0} | Sweep: $SWEEP_ID"
wandb agent --count "$RUNS_PER_AGENT" "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
