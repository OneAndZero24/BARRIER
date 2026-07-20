#!/bin/bash
# ============================================================================
# SLURM – BARRIER Evaluation on ImageNet-Confuse5 (Table 4)
# ============================================================================
# Evaluates a BARRIER-edited model on Confuse5 (5 confusing pairs, 10 concepts).
# Matches ScaPre protocol for Unlearn Acc, Preserve Acc, Overall Acc,
# CLIPcoco, and UQ.
#
# Run:  sbatch SD/scapre/scripts/slurm_eval_confuse5.sh
# ============================================================================

#SBATCH --job-name=eval-confuse5
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=dgxh100

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$HOME/InTAct-Unl:$PYTHONPATH

# ---- Set your checkpoint path below ----
CKPT="models/CONFUSE5_MODEL_PATH/diffusers-CONFUSE5_MODEL_PATH.pt"
cd $(dirname "$0")/../..

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Edit CKPT variable in this script to point to your trained model."
    exit 1
fi

echo "============================================"
echo "  ScaPre Eval – Confuse5 (Table 4)"
echo "  Checkpoint: $CKPT"
echo "============================================"

python scapre/evaluate.py \
    --benchmark confuse5 \
    --ckpt_name "$CKPT" \
    --output_dir results/scapre_eval \
    --coco_prompts_source scapre/datasets/coco_30k.csv \
    --coco_max_images 5000

echo "Done.  Results: results/scapre_eval/confuse5/results_confuse5.json"
