#!/bin/bash
# ============================================================================
# SLURM – DDPM CIFAR-10 InTAct KL Bayesian Sweep (50 runs via wandb agent)
# ============================================================================
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_kl_sweep.sh
# ============================================================================

#SBATCH --job-name=ddpm-kl-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# Redirect caches to avoid home quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

echo "Starting KL bayesian sweep (50 runs) on $(hostname)"

wandb agent $(wandb sweep configs/sweep_kl_bayes.yaml 2>/dev/null | grep -o 'wandb agent .*') --count 50

echo "KL sweep done."
