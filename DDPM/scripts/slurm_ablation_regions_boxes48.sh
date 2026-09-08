#!/bin/bash
# ============================================================================
# SLURM Array Job – Region Ablation, boxes_4 / boxes_8 only
# ============================================================================
# Same protocol as scripts/slurm_ablation_regions.sh but restricted to the two
# predefined m-group constructions:
#   region_mode: boxes_4 | boxes_8
#   lambda:      0.5 1 2 5 10 25
#   seed:        0 1 2
# Grid: 2 variants x 6 lambdas x 3 seeds = 36 jobs
#
# Each job appends a row to r/regions.csv and per-layer diagnostics to
# r/diagnostics.csv (created under DDPM/results/).  After all jobs:
#   python ablation_regions.py --summarize --results_dir /shared/results/common/miksa/intact/DDPM/r
# -> writes r/regions_table.tex
#
# Resources (RTX 4090, 24 GB VRAM):
#   - VRAM  ~2-4 GB training / ~5 GB peak during setup SVD  -> 24 GB is ample
#   - RAM   ~16-20 GB peak (8 attn layers x 1.31 GB forget-activation buffers
#     during setup)  -> 32 GB requested; 16 GB would OOM
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ablation_regions_boxes48.sh
# ============================================================================

#SBATCH --job-name=ablate-boxes48
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-35

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Grid mapping (array index -> variant, lambda, seed)
#   idx = 18*v + 6*lam_idx + seed_idx
# ============================================================================
VARIANTS=(boxes_4 boxes_8)
LAMBDAS=(0.5 1 2 5 10 25)
SEEDS=(0 1 2)

IDX=${SLURM_ARRAY_TASK_ID}
V_IDX=$(( IDX / 18 ))
L_IDX=$(( (IDX % 18) / 3 ))
S_IDX=$(( IDX % 3 ))

VARIANT=${VARIANTS[$V_IDX]}
LAMBDA=${LAMBDAS[$L_IDX]}
SEED=${SEEDS[$S_IDX]}

echo "============================================"
echo "boxes4/8 ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  variant=${VARIANT}  lambda=${LAMBDA}  seed=${SEED}"
echo "============================================"

python ablation_regions.py \
    --region_mode "${VARIANT}" \
    --lambda "${LAMBDA}" \
    --seed "${SEED}" \
    --config configs/pipeline_fulleval.yaml \
    --n_iters 3000 \
    --results_dir /shared/results/common/miksa/intact/DDPM/r

echo "boxes4/8 ablation – ${VARIANT} lam=${LAMBDA} seed=${SEED} – Job ${IDX} complete."