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
#
# NOTE: conda activate hangs on the dgx compute nodes, so this script never
# sources conda - it calls the env's binaries directly via PATH. The sweep
# yaml runs the program through ${env} (the env python), so nothing else
# needs conda either.
# ============================================================================

#SBATCH --job-name=esc-intact-sweep
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=rtx4090
#SBATCH --time=24:00:00
#SBATCH --array=0-3
# Alternatives: --partition=rtx4090_batch --qos=batch (preemptible), or --partition=dgx --qos=big

set -euo pipefail

WANDB_ENTITY="oneandzero24"
WANDB_PROJECT="esc-intact-tinyimagenet"
RUNS_PER_AGENT="${2:-10}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <wandb-sweep-id> [runs-per-agent]"
    exit 1
fi

SWEEP_ID="$1"

ESC_ENV="/shared/results/common/miksa/envs/ESC"
export PATH="$ESC_ENV/bin:$PATH"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

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

echo "Array task ${SLURM_ARRAY_TASK_ID:-0} sweep $SWEEP_ID on $(hostname) $(date '+%F %T')"
"$ESC_ENV/bin/wandb" agent --count "$RUNS_PER_AGENT" "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"