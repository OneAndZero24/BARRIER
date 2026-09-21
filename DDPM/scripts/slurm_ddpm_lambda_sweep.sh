#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM InTAct lambda_interval sweep (CIFAR-10, forget airplane)
# ============================================================================
# Held at the paper's configuration (BARRIER / InTAct):
#   targets  = self-attention QKV projections + class-embedding MLP
#   k        = 32 (reduced_dim), Adam, lr = 1e-4, 3000 steps, RL objective,
#   region   = two_corner (paper baseline), no remain-set loss.
#
# Swept: lambda_interval in {0.01, 0.1, 1, 2, 5, 10, 100} x seeds {0, 1, 2}
#        = 21 jobs (array 0-20).
#
# Each job runs headless (no wandb) via ablation_regions.py and writes a
# per-run sidecar <results_dir>/runs/*/rows.json with UA / RA (ta) / FID.
#
# After the array completes, collect metrics + render the pairwise curves:
#   python scripts/plot_lambda_sweep.py \
#       --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_lambda_sweep.sh
# ============================================================================

#SBATCH --job-name=ddpm-lambda-sweep
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-20

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# Redirect caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

LAMBDAS=(0.01 0.1 1 2 5 10 100)
SEEDS=(0 1 2)
N_SEED=${#SEEDS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LAM_IDX=$((IDX / N_SEED))
SEED_IDX=$((IDX % N_SEED))
LAM=${LAMBDAS[$LAM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

RESULTS_DIR=/shared/results/common/miksa/intact/DDPM/lambda_sweep

echo "============================================"
echo "lambda sweep – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  lambda_interval=${LAM}  seed=${SEED}"
echo "============================================"

python ablation_regions.py \
    --experiment lambda_sweep \
    --region_mode two_corner \
    --interval_mode full \
    --lambda ${LAM} \
    --seed ${SEED} \
    --config configs/pipeline_fulleval.yaml \
    --n_iters 3000 \
    --results_dir ${RESULTS_DIR}

echo "lambda sweep – index ${IDX} (lambda=${LAM}, seed=${SEED}) complete."
