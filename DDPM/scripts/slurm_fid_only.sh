#!/bin/bash
# ============================================================================
# SLURM – FID-Only Evaluation for Pre-Unlearned Checkpoints
# ============================================================================
# Computes FID (5000 samples/class, reference-paper size) for 3 checkpoints:
#   IA NO SVD, SVD NO IA, GA
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_fid_only.sh
# ============================================================================

#SBATCH --job-name=fid-only
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Checkpoints & labels
# ============================================================================
CHECKPOINT_1="/shared/results/common/miksa/intact/DDPM/results/pipeline/2026_08_11_015300"
CHECKPOINT_2="/shared/results/common/miksa/intact/DDPM/results/pipeline/2026_08_11_085659"
CHECKPOINT_3="/shared/results/common/miksa/intact/DDPM/results/pipeline/2026_08_11_063656"

LABEL_1="IA NO SVD"
LABEL_2="SVD NO IA"
LABEL_3="GA"

MODEL_CONFIG="configs/cifar10_intact.yml"
REF_DATASET="/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"
LABEL_TO_FORGET=0
N_SAMPLES=5000

echo "============================================"
echo "FID-Only Eval – Job ${SLURM_JOB_ID}"
echo "  qos=big  mem=128GB  partition=dgxa100"
echo "  n_samples_per_class = ${N_SAMPLES}"
echo "============================================"

# ---- Run 1: IA NO SVD ----
echo ""
echo ">>> Checkpoint 1/3: ${LABEL_1}"
python scripts/fid_eval_only.py \
    --ckpt_dir "${CHECKPOINT_1}" \
    --model_config "${MODEL_CONFIG}" \
    --label_to_forget "${LABEL_TO_FORGET}" \
    --n_samples_per_class "${N_SAMPLES}" \
    --ref_dataset_dir "${REF_DATASET}" \
    --run_label "${LABEL_1}"

# ---- Run 2: SVD NO IA ----
echo ""
echo ">>> Checkpoint 2/3: ${LABEL_2}"
python scripts/fid_eval_only.py \
    --ckpt_dir "${CHECKPOINT_2}" \
    --model_config "${MODEL_CONFIG}" \
    --label_to_forget "${LABEL_TO_FORGET}" \
    --n_samples_per_class "${N_SAMPLES}" \
    --ref_dataset_dir "${REF_DATASET}" \
    --run_label "${LABEL_2}"

# ---- Run 3: GA ----
echo ""
echo ">>> Checkpoint 3/3: ${LABEL_3}"
python scripts/fid_eval_only.py \
    --ckpt_dir "${CHECKPOINT_3}" \
    --model_config "${MODEL_CONFIG}" \
    --label_to_forget "${LABEL_TO_FORGET}" \
    --n_samples_per_class "${N_SAMPLES}" \
    --ref_dataset_dir "${REF_DATASET}" \
    --run_label "${LABEL_3}"

echo ""
echo "============================================"
echo "FID-Only Eval – All 3 checkpoints complete."
echo "============================================"
