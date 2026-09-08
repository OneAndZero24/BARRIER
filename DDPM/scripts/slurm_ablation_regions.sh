#!/bin/bash
# ============================================================================
# SLURM Array Job – Ablation over Protected-Region Constructions (BARRIER)
# ============================================================================
# Runs the full ablation grid for DDPM class unlearning (CIFAR-10, forget
# class 0 / airplane) held at the paper's configuration:
#   targets = QKV attention projections + class-embedding MLP
#   k = 32, Adam, lr = 1e-4, 3000 steps, RL objective, no remain-set loss
#
# Grid: 5 region constructions x 6 lambdas x 3 seeds = 90 jobs
#   region_mode: two_corner | two_random | boxes_4 | boxes_8 | slabs_2k
#   lambda:      0.5 1 2 5 10 25
#   seed:        0 1 2
#
# Each job appends a row to r/regions.csv and per-layer diagnostics to
# r/diagnostics.csv (created under DDPM/results/).  After all jobs:
#   python ablation_regions.py --summarize --results_dir /shared/results/common/miksa/intact/DDPM/r
# -> writes r/regions_table.tex
#
# Resources (RTX 4090, 24 GB VRAM):
#   - VRAM  ~2-4 GB training / ~5 GB peak during setup SVD  -> 24 GB is ample
#   - RAM   the pre-fix setup held all raw activation buffers (~16 GB) while
#     projecting the remain set (~18 GB more) -> OOM at 32 GB (fixed: buffers
#     are now freed per layer and before the remain pass; peak ~20 GB).
#     48 GB requested for headroom.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ablation_regions.sh
# ============================================================================

#SBATCH --job-name=ablate-regions
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-89

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Grid mapping (array index -> variant, lambda, seed)
#   idx = 18*v + 6*lam_idx + seed_idx
# ============================================================================
VARIANTS=(two_corner two_random boxes_4 boxes_8 slabs_2k)
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
echo "Region ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  variant=${VARIANT}  lambda=${LAMBDA}  seed=${SEED}"
echo "============================================"

python ablation_regions.py \
    --region_mode "${VARIANT}" \
    --lambda "${LAMBDA}" \
    --seed "${SEED}" \
    --config configs/pipeline_fulleval.yaml \
    --n_iters 3000 \
    --results_dir /shared/results/common/miksa/intact/DDPM/r

echo "Region ablation – ${VARIANT} lam=${LAMBDA} seed=${SEED} – Job ${IDX} complete."