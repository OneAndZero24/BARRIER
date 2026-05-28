#!/bin/bash
# ============================================================================
# SLURM - BARRIER Table-2 ASR Calculation on DGXA100
# ============================================================================
# Computes Pre-ASR and ASR from vendored UnlearnDiffAtk logs.
# Set ATTACK_ROOT to the attacked run logs and BASELINE_ROOT to the no-attack
# baseline logs before submitting this job.
# ============================================================================

#SBATCH --job-name=barrier-table2-asr
#SBATCH --qos=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=dgxa100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SD_ROOT}"

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"

# ---- Inputs ----
# Attacked UnlearnDiffAtk logs.
ATTACK_ROOT="/shared/results/common/miksa/intact/SD/barrier_nudity_dgxa100/attacks/barrier_nudity/ud_runs"

# Baseline no-attack logs.
BASELINE_ROOT="/shared/results/common/miksa/intact/SD/barrier_nudity_dgxa100/attacks/barrier_nudity/no_attack_runs"

# Optional CSV summary output.
CSV_PATH="/shared/results/common/miksa/intact/SD/barrier_nudity_dgxa100/metrics/asr_summary.csv"

python experiments/table2/calculate_asr.py \
  --root "${ATTACK_ROOT}" \
  --root-no-attack "${BASELINE_ROOT}" \
  --csv-path "${CSV_PATH}"

echo "ASR summary written to: ${CSV_PATH}"