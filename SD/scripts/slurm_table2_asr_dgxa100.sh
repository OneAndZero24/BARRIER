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

# SLURM executes a copied script from /var/spool/slurmd; use submit dir instead.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

if [[ -f "experiments/table2/calculate_asr.py" ]]; then
  SD_ROOT="$PWD"
elif [[ -f "SD/experiments/table2/calculate_asr.py" ]]; then
  SD_ROOT="$PWD/SD"
else
  echo "Could not locate experiments/table2/calculate_asr.py from submit directory: $PWD" >&2
  exit 1
fi

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
# Default points to UD's own logger output path used by run_table2.py.
ATTACK_ROOT=${ATTACK_ROOT:-"${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/src/files/results/text_grad_esd_nudity_classifier"}

# Baseline no-attack logs.
# Set this to the no-attack run root (or a single run directory containing config.json/log.json).
BASELINE_ROOT=${BASELINE_ROOT:-"${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/src/files/results/no_attack_esd_nudity"}

# If baseline logs are missing, generate them automatically with UD no_attack config.
AUTO_GENERATE_BASELINE=${AUTO_GENERATE_BASELINE:-1}
UD_SRC_ROOT="${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/src"
NO_ATTACK_CONFIG=${NO_ATTACK_CONFIG:-"${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/configs/nudity/no_attack_esd_nudity_classifier.json"}
NO_ATTACK_RUN_NAME=${NO_ATTACK_RUN_NAME:-"attack_idx_0"}
NO_ATTACK_ATTACK_IDX=${NO_ATTACK_ATTACK_IDX:-0}

# Optional overrides for baseline generation.
TARGET_CKPT=${TARGET_CKPT:-""}
DATASET_PATH=${DATASET_PATH:-""}

if [[ ! -e "${BASELINE_ROOT}" ]]; then
  if [[ "${AUTO_GENERATE_BASELINE}" == "1" ]]; then
    echo "BASELINE_ROOT missing; generating no-attack baseline logs..."
    cd "${UD_SRC_ROOT}"
    BASELINE_CMD=(
      python execs/attack.py
      --config-file "${NO_ATTACK_CONFIG}"
      --logger.json.root "${BASELINE_ROOT}"
      --logger.name "${NO_ATTACK_RUN_NAME}"
      --attacker.attack_idx "${NO_ATTACK_ATTACK_IDX}"
    )
    if [[ -n "${TARGET_CKPT}" ]]; then
      BASELINE_CMD+=(--task.target_ckpt "${TARGET_CKPT}")
    fi
    if [[ -n "${DATASET_PATH}" ]]; then
      BASELINE_CMD+=(--task.dataset_path "${DATASET_PATH}" --attacker.no_attack.dataset_path "${DATASET_PATH}")
    fi
    "${BASELINE_CMD[@]}"
    cd "${SD_ROOT}"
  else
    echo "BASELINE_ROOT does not exist: ${BASELINE_ROOT}" >&2
    echo "Set AUTO_GENERATE_BASELINE=1 or provide BASELINE_ROOT manually." >&2
    exit 1
  fi
fi

# Optional CSV summary output.
CSV_PATH=${CSV_PATH:-"/shared/results/common/miksa/intact/SD/barrier_nudity_dgxa100/metrics/asr_summary.csv"}

echo "SD_ROOT=${SD_ROOT}"
echo "ATTACK_ROOT=${ATTACK_ROOT}"
echo "BASELINE_ROOT=${BASELINE_ROOT}"
echo "CSV_PATH=${CSV_PATH}"

python experiments/table2/calculate_asr.py \
  --root "${ATTACK_ROOT}" \
  --root-no-attack "${BASELINE_ROOT}" \
  --csv-path "${CSV_PATH}"

echo "ASR summary written to: ${CSV_PATH}"