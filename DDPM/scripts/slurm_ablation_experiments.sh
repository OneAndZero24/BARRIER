#!/bin/bash
# ============================================================================
# SLURM Array Job – Mechanism / Design-Choice Experiments 3-9 (DDPM, CIFAR-10)
# ============================================================================
# DDPM class unlearning held at the paper's configuration (targets = QKV
# attention projections + class-embedding MLP, k = 32, Adam, lr = 1e-4,
# 3000 steps, RL objective, no remain-set loss).
#
# Grid (180 jobs, defined in <repo>/expgrid.py):
#   exp3  uniform-margin control        two_corner      6L x 3s =  18
#   exp4  centre vs width               width/centre    2 x 6L x 3s =  36
#   exp5  protected-region family       env_box         6L x 3s =  18
#   exp6  sign-convention sensitivity   frac 0/.25/.5   3 x 3s  =   9
#   exp7  percentile alpha              alpha 1/5/10    3 x 6L x 3s =  54
#   exp8  delta_b in the interval legs  off/on          2 x 6L x 3s =  36
#   exp9  L_res isolation               3 combos        3 x 3s  =   9
#                                                              total = 180
#   lambda sweep 0.5 1 2 5 10 25; seeds 0 1 2; fixed lambda 5 for exp6/exp9.
#
# Each job:
#   1. appends one row to r/ablations.csv (unified schema)
#   2. saves run artifacts (pca_info.pth + run_meta.json) under r/runs/...
#      for ablation_mechanism.py (Experiments 1-2)
# After the array:
#   python ablation_regions.py --summarize --results_dir r
#                                  -> r/tables/exp*_{backend}.tex
#   python ablation_mechanism.py --results_dir r --backbone ddpm --no_remain
#                                  -> Experiments 1-2 (pass --no_remain to
#                                     skip the Exp2 activation collection)
#
# Resources: rtx4090\_batch partition (qos=batch), 32 GB RAM, one GPU per job.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ablation_experiments.sh
# ============================================================================

#SBATCH --job-name=exp3-9-ddpm
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-179

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

IDX=${SLURM_ARRAY_TASK_ID}
FLAGS=$(python /home/miksa/InTAct-Unl/expgrid.py ddpm ${IDX} 2>&1)
if [ $? -ne 0 ]; then
    echo "grid decode failed: ${FLAGS}"
    exit 2
fi

echo "============================================"
echo "Experiments 3-9 – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  flags: ${FLAGS}"
echo "============================================"

python ablation_regions.py \
    ${FLAGS} \
    --config configs/pipeline_fulleval.yaml \
    --n_iters 3000 \
    --results_dir /shared/results/common/miksa/intact/DDPM/r

echo "Experiments 3-9 – index ${IDX} complete."