#!/bin/bash
# ============================================================================
# SLURM Script – SD Train Artists (Baseline UCE / InTAct Unlearning)
# ============================================================================
# Run UCE baseline or InTAct unlearning for artists.
#
# HOW TO USE:
#   sbatch SD/scripts/slurm_train_artists.sh --concepts "Kelly Mckernan" --concept_type art
#   sbatch SD/scripts/slurm_train_artists.sh --concepts "Kelly Mckernan" --concept_type art --intact --lambda_interval 1.0
# ============================================================================

#SBATCH --job-name=sd-train-artists
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm

# Resolve paths dynamically based on SLURM submission or current directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

if [ -d "SD" ]; then
    cd SD
fi

export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

# Redirect caches to scratch directory (from setup_cache.py candidates)
if [ -n "$SCRATCH" ]; then
    export CACHE_ROOT="$SCRATCH/.cache"
else
    export CACHE_ROOT="/shared/results/common/miksa/intact/SD/.cache"
fi

echo "============================================"
echo "SD Train Artists – Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
echo "  Arguments: $@"
echo "  Directory: $(pwd)"
echo "  PYTHONPATH: $PYTHONPATH"
echo "============================================"

# Run the training script
python train-scripts/train_artists.py "$@"

echo "SD Train Artists Job Complete."
