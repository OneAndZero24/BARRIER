#!/bin/bash
# ============================================================================
# SLURM -- DDPM CIFAR-10 InTAct Ablation: SVD no interval (50 runs)
# ============================================================================
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_svd_noia_sweep.sh
# ============================================================================

#SBATCH --job-name=ddpm-svd-noia
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=dgxa100

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

echo "Starting SVD-noIA ablation sweep (50 runs) on $(hostname)"

SWEEP_OUT=$(wandb sweep configs/sweep_svd_noia.yaml 2>&1)
echo "$SWEEP_OUT"
SWEEP_ID=$(echo "$SWEEP_OUT" | grep -oP 'wandb agent \K\S+')
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: could not parse sweep ID"
    exit 1
fi
echo "Sweep ID: $SWEEP_ID"

wandb agent "$SWEEP_ID" --count 50

echo "SVD-noIA sweep done."
