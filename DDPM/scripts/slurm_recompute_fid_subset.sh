#!/bin/bash
# ============================================================================
# SLURM job – recompute FID on 500/class subsets of the saved lambda-sweep
# fid samples (35 runs), using the original TF Inception-V3 evaluator.
# No training / no re-sampling - only reads the saved PNGs.
#
# Writes per-run rows.json["fid_500"]; then aggregate:
#   python scripts/recompute_fid_subset.py --results_dir <RESULTS_DIR> --collect
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_recompute_fid_subset.sh
# ============================================================================

#SBATCH --job-name=fid500-grid
#SBATCH --output=slurm-%j.out
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --requeue

set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/shared/results/common/miksa/intact/DDPM/lambda_sweep}
N_PER_CLASS=${N_PER_CLASS:-500}
METHOD=${METHOD:-block}
SEED=${SEED:-0}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-salun-ddpm}
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"

python scripts/recompute_fid_subset.py \
    --results_dir "${RESULTS_DIR}" \
    -n "${N_PER_CLASS}" \
    --method "${METHOD}" --seed "${SEED}"

echo "done - now run: python scripts/recompute_fid_subset.py --results_dir ${RESULTS_DIR} --collect"