#!/bin/bash
# ============================================================================
# SLURM job – fine-tune ResNet-34 on native 32x32 CIFAR-10 (no 224 upscaling),
# then evaluate remain accuracy (TA) of every lambda-sweep run at 32x32.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_rn34_32_train_eval.sh
# ============================================================================
#SBATCH --job-name=rn34_32
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx2080,dgx,dgxa100
#SBATCH -o /shared/results/common/miksa/intact/DDPM/lambda_sweep/rn34_32.%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm

cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

B=/shared/results/common/miksa/intact/DDPM
CKPT=$B/models/cifar10_resnet34_32.pth

python train_classifier.py --dataset cifar10 --data_path ../data \
  --img_size 32 --lr 1e-4 --n_epochs 30 --out $CKPT

python scripts/eval_remain_32.py \
  --runs_dir $B/lambda_sweep/runs \
  --classifier_ckpt $CKPT \
  --label_to_forget 0 \
  --out_csv $B/lambda_sweep/remain_acc_32.csv
