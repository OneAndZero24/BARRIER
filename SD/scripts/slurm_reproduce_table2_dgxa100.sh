#!/bin/bash
# ============================================================================
# SLURM - Reproduce Table-2 (full pipeline) on DGXA100
#
# Single sbatch to run BARRIER export -> STEREO -> external attacks -> ASR
# and aggregate results into a final CSV/LaTeX-ready table.
#
# Usage (example):
#   sbatch SD/scripts/slurm_reproduce_table2_dgxa100.sh
# Environment overrides:
#   METHODS (space-separated, default: "barrier esd uce concept-ablation")
#   SEEDS (space-separated, default: "0")
#   CHECKPOINT (path to single checkpoint used for all methods) OR
#   CHECKPOINT_DIR (directory containing per-method checkpoints named <method>.pt)
#   EXTERNAL_ATTACKS (comma-separated, default: "ud,cce")
#   OUTPUT_ROOT (where to write results)
# ============================================================================

#SBATCH --job-name=barrier-table2-reproduce
#SBATCH --qos=big
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

# SLURM executes a copied script from /var/spool/slurmd; use submit dir instead.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

if [[ -f "experiments/table2/run_table2.py" ]]; then
  SD_ROOT="$PWD"
elif [[ -f "SD/experiments/table2/run_table2.py" ]]; then
  SD_ROOT="$PWD/SD"
else
  echo "Could not locate experiments/table2/run_table2.py from submit directory: $PWD" >&2
  exit 1
fi

cd "${SD_ROOT}"

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME="/shared/results/common/${USER:-user}/.cache/huggingface"
export TORCH_HOME="/shared/results/common/${USER:-user}/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/${USER:-user}/.cache"

# ---- Defaults / inputs ----
METHODS=${METHODS:-"barrier esd uce concept-ablation"}
SEEDS=${SEEDS:-"0"}
EXTERNAL_ATTACKS=${EXTERNAL_ATTACKS:-"ud,cce"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/shared/results/common/${USER:-user}/intact/SD/reproduce_table2_$(date +%Y%m%d_%H%M%S)"}

# Checkpoint selection: prefer explicit CHECKPOINT, else method-specific files under CHECKPOINT_DIR
CHECKPOINT=${CHECKPOINT:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}

# Install runtime deps if missing (fast, idempotent)
python - <<'PY'
import importlib,sys,subprocess
reqs = ['fastargs','torchmetrics','nudenet','onnxruntime']
for r in reqs:
    try:
        importlib.import_module(r)
    except Exception:
        subprocess.check_call([sys.executable,'-m','pip','install',r])
PY

mkdir -p "${OUTPUT_ROOT}"

echo "SD_ROOT=${SD_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "METHODS=${METHODS}"
echo "SEEDS=${SEEDS}"
echo "EXTERNAL_ATTACKS=${EXTERNAL_ATTACKS}"

# Loop methods and seeds
for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    RUN_DIR="${OUTPUT_ROOT}/${method}_seed_${seed}"
    mkdir -p "${RUN_DIR}"

    # Determine checkpoint for this method
    if [[ -n "${CHECKPOINT}" ]]; then
      CKPT="${CHECKPOINT}"
    elif [[ -n "${CHECKPOINT_DIR}" && -f "${CHECKPOINT_DIR}/${method}.pt" ]]; then
      CKPT="${CHECKPOINT_DIR}/${method}.pt"
    else
      echo "No checkpoint provided for method ${method}. Provide CHECKPOINT or CHECKPOINT_DIR with ${method}.pt" >&2
      exit 1
    fi

    echo "Running method=${method} seed=${seed} checkpoint=${CKPT} -> ${RUN_DIR}"

    python experiments/table2/run_table2.py \
      --method "${method}" \
      --checkpoint "${CKPT}" \
      --concept "nudity" \
      --output_dir "${RUN_DIR}" \
      --device "cuda" \
      --external_attacks "${EXTERNAL_ATTACKS}" \
      --attack_idx "${seed}"

    # Compute ASR for UD (if present)
    ATTACK_ROOT="${RUN_DIR}/attacks/${method}_nudity/ud_logs"
    BASELINE_ROOT="${RUN_DIR}/attacks/${method}_nudity/ud_no_attack_logs"
    CSV_PATH="${RUN_DIR}/metrics/asr_summary.csv"

    if [[ -d "${ATTACK_ROOT}" ]]; then
      echo "Computing ASR for: ${method} seed ${seed}"
      python experiments/table2/calculate_asr.py \
        --root "${ATTACK_ROOT}" \
        --root-no-attack "${BASELINE_ROOT}" \
        --csv-path "${CSV_PATH}" || echo "ASR computation failed for ${RUN_DIR}" >&2
    else
      echo "No UD attack logs at ${ATTACK_ROOT}; skipping ASR" >&2
    fi
  done
done

# Aggregate all per-run ASR results into final Table-2 CSV + LaTeX
python experiments/table2/aggregate_table2.py --root "${OUTPUT_ROOT}" --out-csv "${OUTPUT_ROOT}/metrics/table2_final.csv" --out-latex "${OUTPUT_ROOT}/metrics/table2_final.tex"

echo "Reproduction finished. Results under: ${OUTPUT_ROOT}/metrics"
