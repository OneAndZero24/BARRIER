#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM InTAct lambda sweep, extra pareto points
# ============================================================================
# 1) Rerun the existing pareto lambdas at the canonical seed 1234 (same seed
#    used for the original DDPM CIFAR-10 result in the paper table):
#        lambda in {0.01, 0.1, 1, 5, 10, 100}  x  seed 1234
# 2) Add the two new lambdas at all 4 seeds {0,1,2,1234}:
#        lambda in {30, 50}  x  seeds {0, 1, 2, 1234}
#    => 6 + 8 = 14 jobs (array 0-13).
#
# Same fixed paper config as slurm_ddpm_lambda_sweep.sh (QKV + class-embed
# targets, k=32, Adam lr=1e-4, 3000 steps, RL, two_corner, interval full).
# Writes sidecars into the same <results_dir>/runs so each lambda ends up
# with 4 seeds {0,1,2,1234}.
#
# After completion, plot (auto pareto-front ordering):
#   python scripts/plot_lambda_sweep.py \
#       --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_lambda_sweep_extra.sh
# ============================================================================

#SBATCH --job-name=ddpm-lambda-extra
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-13

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

# Redirect caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

CANONICAL_SEED=1234
LAMBDAS_ONE=(0.01 0.1 1 5 10 100)     # rerun at seed 1234
LAMBDAS_FOUR=(30 50)                   # all 4 seeds
SEEDS_FOUR=(0 1 2 1234)

IDX=${SLURM_ARRAY_TASK_ID}
if [ ${IDX} -lt 6 ]; then
    LAM=${LAMBDAS_ONE[${IDX}]}
    SEED=${CANONICAL_SEED}
else
    J=$((IDX - 6))
    LAM=${LAMBDAS_FOUR[$((J / 4))]}
    SEED=${SEEDS_FOUR[$((J % 4))]}
fi

RESULTS_DIR=/shared/results/common/miksa/intact/DDPM/lambda_sweep

echo "============================================"
echo "extra lambda sweep – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
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

echo "lambda sweep (extra) – index ${IDX} (lambda=${LAM}, seed=${SEED}) complete."