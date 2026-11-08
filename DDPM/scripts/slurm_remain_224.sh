#!/bin/bash
# ============================================================================
# SLURM job – remain accuracy (TA) + forgotten-class (airplane) rate of every
# lambda-sweep run with the ORIGINAL 224-trained ResNet-34 (SEMU protocol:
# 32x32 samples upscaled to 224x224). Companion to slurm_rn34_32_train_eval.sh.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_remain_224.sh
# ============================================================================
#SBATCH --job-name=remain224
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx2080,dgx,dgxa100
#SBATCH -o /shared/results/common/miksa/intact/DDPM/lambda_sweep/remain224.%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm

cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

B=/shared/results/common/miksa/intact/DDPM

python scripts/eval_remain_32.py \
  --runs_dir $B/lambda_sweep/runs \
  --classifier_ckpt $B/models/cifar10_resnet34.pth \
  --img_size 224 \
  --label_to_forget 0 \
  --out_csv $B/lambda_sweep/remain_acc_224.csv
