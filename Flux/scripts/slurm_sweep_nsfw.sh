#!/bin/bash
# ============================================================================
# SLURM Job – NSFW Sweep (Flux)
# ============================================================================
# Runs a full wandb sweep for NSFW erasure on Flux with H100 / 256 GB nodes.
# The sweep configuration lives at `configs/intact/sweep_nsfw.yaml`.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_sweep_nsfw.sh
# ============================================================================

#SBATCH --job-name=flux-nsfw-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux          # adjust to the appropriate env if needed
cd $HOME/InTAct-Unl/Flux
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# caches to shared storage
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"
export WANDB_CACHE_DIR="/shared/results/common/miksa/.cache/wandb"
export CLIP_CACHE_DIR="/shared/results/common/miksa/.cache/clip"

echo "Starting Flux NSFW sweep on $(hostname)"

# Launch the sweep (this script will run until the agent exits).
./run_sweep.sh sweep_nsfw

echo "Flux NSFW sweep finished."
