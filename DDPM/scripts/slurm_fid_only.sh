#!/bin/bash
# ============================================================================
# FID-only backfill for DDPM region-grid runs killed before FID was computed
# (two_corner / two_random at lambda=0.5 in the region grid).  Reuses the
# saved fid_samples PNGs -- no training, no sampling.
#
# FID = torchmetrics InceptionV3 pool3 (2048-dim), the same statistics as the
# paper's TF evaluator.  Patches rows.json sidecars; then run:
#   python ablation_regions.py --summarize --
#       --results_dir /shared/results/common/miksa/intact/DDPM/r
#
# Usage:  sbatch scripts/slurm_fid_only.sh
# ============================================================================

#SBATCH --job-name=fid-backfill
#SBATCH --partition=rtx3080
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

python fid_only.py \
    --results_dir /shared/results/common/miksa/intact/DDPM/r \
    --region_modes two_corner two_random \
    --lambda 0.5 \
    --n 2048

echo "FID backfill complete."