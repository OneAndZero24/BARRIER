#!/bin/bash
# ============================================================================
# SLURM Array - ESC/InTAct Tiny-ImageNet ViT sweep agents (wandb)
# ============================================================================
# Two-step usage (identical to SD/scripts/slurm_artist_sweep_array.sh):
#   Step 1 (on login node):
#     cd ~/InTAct-Unl/Classification/ESC
#     wandb sweep --project esc-intact-tinyimagenet --entity oneandzero24 \
#         configs/sweep_bayes_intact_tinyimagenet.yaml
#     # note the sweep id from the output, e.g. "abcd1234"
#
#   Step 2:
#     sbatch --array=0-3 slurm_wandb_sweep.sh <SWEEP_ID> [runs-per-agent]
#
#   ESC baseline grid:
#     wandb sweep --project esc-intact-tinyimagenet --entity oneandzero24 \
#         configs/sweep_grid_esc_tinyimagenet.yaml
#     sbatch --array=0-3 slurm_wandb_sweep.sh <SWEEP_ID> 2
# ============================================================================

#SBATCH --job-name=esc-intact-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --array=0-3

set -euo pipefail

WANDB_ENTITY="oneandzero24"
WANDB_PROJECT="esc-intact-tinyimagenet"
RUNS_PER_AGENT="${2:-10}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <wandb-sweep-id> [runs-per-agent]"
    exit 1
fi

SWEEP_ID="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/ESC

# ---- Cache redirects (never in $HOME) ----
export CACHE_ROOT="/shared/results/common/miksa/esc-intact/.cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

cd "$HOME/InTAct-Unl/Classification/ESC"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

echo "Array task: ${SLURM_ARRAY_TASK_ID:-0} | Sweep: $SWEEP_ID"
wandb agent --count "$RUNS_PER_AGENT" "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
