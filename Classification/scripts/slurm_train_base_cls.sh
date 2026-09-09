#!/bin/bash
# ============================================================================
# Train the base ResNet-18 (CIFAR-10) checkpoint for the mechanism/design-choice
# ablations.  SalUn protocol: SGD lr=0.1, 182 epochs, decay at 91/136, seed 1.
# Saves <models_dir>/0checkpoint.pth.tar  (pruning=0 + filename convention).
#
# Usage:  sbatch scripts/slurm_train_base_cls.sh
# ============================================================================

#SBATCH --job-name=train-base-rn18
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

MODELS_DIR=/shared/results/common/miksa/intact/Cls/models

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/Classification
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

mkdir -p ${MODELS_DIR}

python main_train.py \
    --data /home/miksa/InTAct-Unl/data \
    --seed 1 --train_seed 1 \
    --save_dir ${MODELS_DIR}

echo "Base ResNet-18 training complete: ${MODELS_DIR}/0checkpoint.pth.tar"