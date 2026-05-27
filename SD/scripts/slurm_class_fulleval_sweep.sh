#!/bin/bash
# ============================================================================
# SLURM launcher for the SD class-forgetting W&B sweep
# ============================================================================
# Usage:
#   cd SD
#   sbatch scripts/slurm_class_fulleval_sweep.sh <wandb-sweep-id>
# ============================================================================

#SBATCH --job-name=sd-class-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <wandb-sweep-id>"
  exit 1
fi

SWEEP_ID="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

wandb agent "$SWEEP_ID"