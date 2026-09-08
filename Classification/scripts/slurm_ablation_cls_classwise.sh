#!/bin/bash
# ============================================================================
# SLURM Array Job – Mechanism / Design-Choice Experiments 3-9 (ResNet-18,
# CIFAR-10, CLASSWISE forgetting: class 0 / airplane)
# ============================================================================
# Protocol: target layer = final FC, k = 32, SGD lr = 1e-3, 10 epochs, RL
# objective.  Fixed (paper) lambda = 10; sweep 0.5 1 2 5 10 25; seeds 0..2.
#
# Grid (180 jobs, defined in <repo>/expgrid.py as "resnet18-classwise"):
#   exp3 18 | exp4 36 | exp5 18 | exp6 9 | exp7 54 | exp8 36 | exp9 9
#
# Each job appends one row to
#   /shared/results/common/miksa/intact/Cls/r/ablations.csv
# and saves run artifacts (pca_info.pth + ckpt.pth + run_meta.json) under
#   .../Cls/r/runs/<suffix>/   for ablation_mechanism.py (Experiments 1-2).
#
# After the array:
#   python ablation_cls.py --summarize \
#       --results_dir /shared/results/common/miksa/intact/Cls/r
#   python ablation_mechanism.py --results_dir ... \
#       --backbone resnet18 --region_mode two_corner --experiment exp3
#
# Usage:
#   cd Classification
#   sbatch scripts/slurm_ablation_cls_classwise.sh
# ============================================================================

#SBATCH --job-name=intact-exp3-9-cls-cw
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-179

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/Classification
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

IDX=${SLURM_ARRAY_TASK_ID}
FLAGS=$(python /home/miksa/InTAct-Unl/expgrid.py resnet18-classwise ${IDX} 2>&1)
if [ $? -ne 0 ]; then
    echo "grid decode failed: ${FLAGS}"
    exit 2
fi

echo "============================================"
echo "Experiments 3-9 (classwise) – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  flags: ${FLAGS}"
echo "============================================"

python ablation_cls.py \
    ${FLAGS} \
    --config configs/pipeline_classwise.yaml \
    --results_dir /shared/results/common/miksa/intact/Cls/r

echo "Experiments 3-9 (classwise) – index ${IDX} complete."