#!/bin/bash
# ============================================================================
# SLURM – BARRIER Training on ImageNet-Diversi50 (50 concepts)
# ============================================================================
# Trains BARRIER/InTAct on all 50 Diversi50 concepts using ImageNet-1K
# training images. Produces a diffusers UNet checkpoint.
#
# Run:  sbatch SD/scapre/scripts/slurm_train_diversi50.sh
# ============================================================================

#SBATCH --job-name=barrier-diversi50
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=dgxh100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD

export PYTHONPATH=$HOME/InTAct-Unl:$PYTHONPATH

# ---- Training ----
python scapre/train.py \
    --benchmark diversi50 \
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
    --bounds_fraction 0.3

echo "Training complete.  Evaluate with:"
echo "  python scapre/evaluate.py --benchmark diversi50 --ckpt_name <diffusers-pt-path> --output_dir results/diversi50"
