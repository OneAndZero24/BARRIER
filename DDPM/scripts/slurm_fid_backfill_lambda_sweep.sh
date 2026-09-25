#!/bin/bash
# ============================================================================
# FID backfill (GPU, RTX 4090) for the ddpm-fid-repro lambda sweep
# ============================================================================
# Recomputes FID with the ORIGINAL TF Inception-V3 evaluator (evaluator.py,
# pool_3 2048-dim — same code path as compute_fid_full.py) for any run whose
# rows.json sidecar has no numeric FID, then patches the sidecar.
#
# Requires tensorflow -> conda env salun-ddpm (matches DDPM/requirements.txt).
#
# After backfill, render the pareto curves:
#   python scripts/plot_lambda_sweep.py \
#       --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_fid_backfill_lambda_sweep.sh
# ============================================================================

#SBATCH --job-name=fid-backfill-lambda
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail

# Same GPU guard as slurm_ddpm_lambda_sweep_fid.sh (rtx4090_batch also
# schedules RTX 5090 nodes; salun-ddpm's torch/tf have no sm_120 support).
MAX_REQUEUE=${MAX_REQUEUE:-5}
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: ${GPU_NAME}  (compute capability ${GPU_CAP:-unknown})"
if [ -n "${GPU_CAP}" ] && [ "${GPU_CAP}" -ge 10 ]; then
    ATTEMPT=${SLURM_RESTART_COUNT:-0}
    echo "FATAL-REQUEUE: capability ${GPU_CAP}.x (Blackwell sm_120+) unsupported"
    if [ "${ATTEMPT}" -lt "${MAX_REQUEUE}" ]; then
        echo "requeueing (attempt $((ATTEMPT + 1))/${MAX_REQUEUE}) ..."
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-salun-ddpm}
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

python scripts/fid_backfill_lambda_sweep.py \
    --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep

echo "FID backfill complete. Run plot_lambda_sweep.py next."