#!/bin/bash
# ============================================================================
# SLURM – BARRIER Evaluation on ImageNet-Diversi50 (Table 3)
# ============================================================================
# Evaluates a BARRIER-edited model on all 50 Diversi50 concepts plus
# COCO CLIP score.  Matches ScaPre protocol EXACTLY:
#   - SD v1.5 pipeline (runwayml/stable-diffusion-v1-5)
#   - PNDM scheduler, 50 steps, cfg=7.5, 512x512
#   - ResNet-50 (ImageNet weights) with substring matching
#   - CLIP ViT-B/32 for CLIPcoco
#   - Same seeds from imagenet-50.csv
#
# Run:  sbatch SD/scapre/scripts/slurm_eval_diversi50.sh
# ============================================================================

#SBATCH --job-name=eval-diversi50
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
CKPT="models/DIVERSI50_MODEL_PATH/diffusers-DIVERSI50_MODEL_PATH.pt"
cd $(dirname "$0")/../..

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "Edit CKPT variable in this script to point to your trained model."
    exit 1
fi

echo "============================================"
echo "  ScaPre Eval – Diversi50 (Table 3)"
echo "  Checkpoint: $CKPT"
echo "============================================"

python scapre/evaluate.py \
    --benchmark diversi50 \
    --ckpt_name "$CKPT" \
    --output_dir results/scapre_eval \
    --coco_prompts_source scapre/datasets/coco_30k.csv \
    --coco_max_images 5000

echo "Done.  Results: results/scapre_eval/diversi50/results_diversi50.json"
