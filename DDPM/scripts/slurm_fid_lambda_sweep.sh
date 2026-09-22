#!/bin/bash
# ============================================================================
# FID-only backfill (GPU) for the DDPM InTAct lambda_interval sweep.
# Reuses the ALREADY-SAVED fid_samples PNGs — no training, no re-sampling.
#
# The main runs produced FID=NaN because the salun-ddpm2 env has no
# TensorFlow and the torchmetrics fallback had a CPU/GPU device mismatch.
# This backfills FID with the EXACT table-scale evaluator (evaluator.py
# Inception graph, pool_3 2048-dim) and patches each run's rows.json sidecar.
# Requires: pip install tensorflow   (in the salun-ddpm2 env).
#
# Then run the plot script:
#   python scripts/plot_lambda_sweep.py \
#       --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
#
# Usage:  sbatch scripts/slurm_fid_lambda_sweep.sh
# ============================================================================

#SBATCH --job-name=fid-lambda-sweep
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --time=02:00:00

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

python fid_only.py \
    --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep \
    --experiment lambda_sweep \
    --region_modes two_corner \
    --backend tf \
    --force

echo "FID backfill complete. Run plot_lambda_sweep.py next."
