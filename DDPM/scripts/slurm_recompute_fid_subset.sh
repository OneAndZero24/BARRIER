#!/bin/bash
# ============================================================================
# SLURM job – recompute FID on N/class subsets of the saved lambda-sweep
# fid samples (35 runs), using the original TF Inception-V3 evaluator.
# No training / no re-sampling - only reads the saved PNGs.
#
# Modes:
#   default           : sequential over all runs (N_PER_CLASS=500 block)
#   ARRAY=1           : one task per run (sbatch --array=0-34), each ~1h
#   REF_STATS_ONLY=1  : just compute+cache reference activations, then exit
#                       (run once first when using a NEW reference dataset)
#
# Writes per-run rows.json["fid_<N>"]; then aggregate:
#   python scripts/recompute_fid_subset.py --results_dir <RESULTS_DIR> --collect
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_recompute_fid_subset.sh
#   REF_DIR=/shared/.../cifar10_without_label_0_n5000 N_PER_CLASS=5000 ARRAY=1 \
#       sbatch --array=0-34 scripts/slurm_recompute_fid_subset.sh
# ============================================================================

#SBATCH --job-name=fidN-grid
#SBATCH --output=slurm-%A_%a.out
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --requeue

set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/shared/results/common/miksa/intact/DDPM/lambda_sweep}
REF_DIR=${REF_DIR:-/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0}
N_PER_CLASS=${N_PER_CLASS:-500}
METHOD=${METHOD:-block}
SEED=${SEED:-0}
ARRAY=${ARRAY:-0}
REF_STATS_ONLY=${REF_STATS_ONLY:-0}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-salun-ddpm}
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"

if [ "${REF_STATS_ONLY}" = "1" ]; then
    python scripts/recompute_fid_subset.py \
        --results_dir "${RESULTS_DIR}" --ref_dir "${REF_DIR}" --ref-stats-only
    echo "reference stats cached - ready for the sweep."
    exit 0
fi

if [ "${ARRAY}" = "1" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUNS=($(ls -d "${RESULTS_DIR}"/runs/lam*_seed* 2>/dev/null | sort))
    IDX=${SLURM_ARRAY_TASK_ID}
    if [ -z "${RUNS[${IDX}]:-}" ]; then
        echo "no run dir for index ${IDX} (found ${#RUNS[@]} runs)"
        exit 1
    fi
    echo "array task ${IDX}: ${RUNS[${IDX}]}"
    python scripts/recompute_fid_subset.py \
        --results_dir "${RESULTS_DIR}" \
        --ref_dir "${REF_DIR}" \
        -n "${N_PER_CLASS}" \
        --method "${METHOD}" --seed "${SEED}" \
        --run-dir "${RUNS[${IDX}]}"
    echo "done - now run: python scripts/recompute_fid_subset.py --results_dir ${RESULTS_DIR} --collect"
    exit 0
fi

python scripts/recompute_fid_subset.py \
    --results_dir "${RESULTS_DIR}" \
    --ref_dir "${REF_DIR}" \
    -n "${N_PER_CLASS}" \
    --method "${METHOD}" --seed "${SEED}"

echo "done - now run: python scripts/recompute_fid_subset.py --results_dir ${RESULTS_DIR} --collect"