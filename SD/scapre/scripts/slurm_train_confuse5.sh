#!/bin/bash
# ============================================================================
# SLURM – BARRIER Training on ImageNet-Confuse5 (10 concepts, 5 pairs)
# ============================================================================

#SBATCH --job-name=barrier-confuse5
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=dgxh100

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$HOME/InTAct-Unl:$PYTHONPATH

python scapre/train.py \
    --benchmark confuse5 \
    --imagenet_root /datasets/ImageNet \
    --base_method rl \
    --lr 5e-6 \
    --epochs 5 \
    --batch_size 8 \
    --targets to_q to_k to_v \
    --lambda_interval 4.0 \
    --reduced_dim 32 \
    --infinity_scale 18.0 \
    --use_actual_bounds \
    --bounds_fraction 0.5

echo "Training complete."
