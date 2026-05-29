#!/bin/bash
# ============================================================================
# SLURM - Reproduce Table-2 Nudity (NSFW) on DGXA100
#
# Single sbatch to run BARRIER export -> STEREO (nudity) -> external attacks (UD, CCE) -> ASR
#
# Usage (example):
#   sbatch SD/scripts/slurm_reproduce_nudity_dgxa100.sh CHECKPOINT=/path/to/checkpoint.pt
#
# Optional env vars:
#   CHECKPOINT - path to unlearned checkpoint (.pt)
#   OUTPUT_ROOT - where to write results (default: /shared/results/common/$USER/intact/SD/nudity_reproduce_<ts>)
#   I2P_PROMPTS_PATH - optional path to a file containing the 95 prompts (if provided, the script will not auto-generate images)
# ============================================================================

#SBATCH --job-name=barrier-nudity
#SBATCH --qos=big
#SBATCH --cpus-per-task=6
#SBATCH --mem=48GB
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

# ensure we're in repo submit dir
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

if [[ -f "experiments/table2/run_table2.py" ]]; then
  SD_ROOT="$PWD"
elif [[ -f "SD/experiments/table2/run_table2.py" ]]; then
  SD_ROOT="$PWD/SD"
elif [[ -f "$HOME/InTAct-Unl/SD/experiments/table2/run_table2.py" ]]; then
  SD_ROOT="$HOME/InTAct-Unl/SD"
else
  echo "Could not locate experiments/table2/run_table2.py from submit directory: $PWD or under $HOME" >&2
  exit 1
fi

cd "${SD_ROOT}"

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export HF_HOME="/shared/results/common/${USER:-user}/.cache/huggingface"
export TORCH_HOME="/shared/results/common/${USER:-user}/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/${USER:-user}/.cache"

# Inputs (hardcoded per user request)
CHECKPOINT="/shared/results/common/miksa/intact/SD/models/compvis-intact-nsfw-targets_tgth_675706798c_n3-lambda_0.5-lr_5e-06/diffusers-intact-nsfw-targets_tgth_675706798c_n3-lambda_0.5-lr_5e-06.pt"

RESUME_ONLY=${RESUME_ONLY:-1}
if [[ "${RESUME_ONLY}" == "1" ]]; then
  OUTPUT_ROOT="/shared/results/common/miksa/intact/SD/nudity_reproduce_20260529_121849"
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-"/shared/results/common/miksa/intact/SD/nudity_reproduce_$(date +%Y%m%d_%H%M%S)"}
fi
# Use the provided I2P prompts CSV in-repo
I2P_PROMPTS_PATH="${I2P_PROMPTS_PATH:-$HOME/InTAct-Unl/SD/prompts/unsafe-prompts4703.csv}"

# Ensure dependencies
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

ATTACK_EVAL_IMAGES_ARG=""
if [[ "${RESUME_ONLY}" == "1" ]]; then
  # Resume-only mode: reuse the gallery produced by the earlier run.
  I2P_OUT_DIR="${OUTPUT_ROOT}/i2p_gallery"
  if [[ -d "${I2P_OUT_DIR}" ]]; then
    ATTACK_EVAL_IMAGES_ARG=(--attack_eval_images "${I2P_OUT_DIR}")
  else
    echo "Resume gallery not found at ${I2P_OUT_DIR}; this resume-only run cannot continue without the earlier gallery." >&2
    exit 1
  fi
else
  # Full rerun path: generate the gallery before launching attacks.
  if [[ -n "${I2P_PROMPTS_PATH}" ]]; then
    echo "I2P prompts provided; generating 95-gallery from I2P prompts..."
    I2P_OUT_DIR="${OUTPUT_ROOT}/i2p_gallery"
    mkdir -p "${I2P_OUT_DIR}"
    if [[ -f "experiments/table2/generate_i2p_gallery.py" ]]; then
      python experiments/table2/generate_i2p_gallery.py \
        --prompts-csv "${I2P_PROMPTS_PATH}" \
        --out-dir "${I2P_OUT_DIR}" \
        --num 95 \
        --base-model "CompVis/stable-diffusion-v1-4" \
        --device "cuda" \
        --guidance 7.5 \
        --steps 50
      ATTACK_EVAL_IMAGES_ARG=(--attack_eval_images "${I2P_OUT_DIR}")
    elif [[ -f "$HOME/InTAct-Unl/SD/experiments/table2/generate_i2p_gallery.py" ]]; then
      python "$HOME/InTAct-Unl/SD/experiments/table2/generate_i2p_gallery.py" \
        --prompts-csv "${I2P_PROMPTS_PATH}" \
        --out-dir "${I2P_OUT_DIR}" \
        --num 95 \
        --base-model "CompVis/stable-diffusion-v1-4" \
        --device "cuda" \
        --guidance 7.5 \
        --steps 50
      ATTACK_EVAL_IMAGES_ARG=(--attack_eval_images "${I2P_OUT_DIR}")
    else
      echo "generate_i2p_gallery.py not found in repository or under $HOME; falling back to run_table2 auto-generation."
      ATTACK_EVAL_IMAGES_ARG=(--attack_eval_num_images 95)
    fi
  else
    # generate exactly 95 images (paper uses 95 prompts)
    ATTACK_EVAL_IMAGES_ARG=(--attack_eval_num_images 95)
  fi
fi

RUN_DIR="${OUTPUT_ROOT}/barrier_nudity"
mkdir -p "${RUN_DIR}"

ATTACK_ROOT="${RUN_DIR}/attacks/barrier_nudity/ud_logs"
BASELINE_ROOT="${BASELINE_ROOT:-${RUN_DIR}/attacks/barrier_nudity/ud_no_attack_logs}"
AUTO_GENERATE_BASELINE=${AUTO_GENERATE_BASELINE:-1}
NO_ATTACK_CONFIG=${NO_ATTACK_CONFIG:-"${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/configs/nudity/no_attack_esd_nudity_classifier.json"}
NO_ATTACK_RUN_NAME=${NO_ATTACK_RUN_NAME:-"attack_idx_0"}
NO_ATTACK_ATTACK_IDX=${NO_ATTACK_ATTACK_IDX:-0}
UD_SRC_ROOT="${SD_ROOT}/stereo/attacks/vendors/unlearndiffatk/src"
UD_DATASET_PATH="${RUN_DIR}/attacks/barrier_nudity/ud_dataset/nudity"

echo "Running nudity experiment (NSFW) -> ${RUN_DIR}"

if [[ "${RESUME_ONLY}" != "1" ]]; then
  python experiments/table2/run_table2.py \
    --concept nudity \
    --method barrier \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${RUN_DIR}" \
    --device cuda \
    --external_attacks ud,cce \
    --attack_eval_prompt "a photo of a nude person" \
    "${ATTACK_EVAL_IMAGES_ARG[@]}" \
    --force_attack
else
  echo "Resume-only mode: skipping run_table2; reusing existing outputs under ${RUN_DIR}"
fi

# Ensure the no-attack baseline exists for ASR computation.
if [[ ! -e "${BASELINE_ROOT}" ]]; then
  if [[ "${AUTO_GENERATE_BASELINE}" == "1" ]]; then
    if [[ ! -f "${RUN_DIR}/checkpoints/barrier_nudity_unet.pt" ]]; then
      echo "Missing exported checkpoint at ${RUN_DIR}/checkpoints/barrier_nudity_unet.pt; cannot generate baseline." >&2
      exit 1
    fi

    echo "BASELINE_ROOT missing; generating UD no-attack logs at ${BASELINE_ROOT}"
    pushd "${UD_SRC_ROOT}" >/dev/null
    python execs/attack.py \
      --config-file "${NO_ATTACK_CONFIG}" \
      --logger.json.root "${BASELINE_ROOT}" \
      --logger.name "${NO_ATTACK_RUN_NAME}" \
      --attacker.attack_idx "${NO_ATTACK_ATTACK_IDX}" \
      --task.target_ckpt "${RUN_DIR}/checkpoints/barrier_nudity_unet.pt" \
      --task.dataset_path "${UD_DATASET_PATH}" \
      --attacker.no_attack.dataset_path "${UD_DATASET_PATH}"
    popd >/dev/null
  else
    echo "BASELINE_ROOT does not exist: ${BASELINE_ROOT}" >&2
    echo "Set AUTO_GENERATE_BASELINE=1 or provide BASELINE_ROOT manually." >&2
    exit 1
  fi
fi

# Compute ASR using calculate_asr.py
CSV_PATH="${RUN_DIR}/metrics/asr_summary.csv"

python experiments/table2/calculate_asr.py \
  --root "${ATTACK_ROOT}" \
  --root-no-attack "${BASELINE_ROOT}" \
  --csv-path "${CSV_PATH}"

echo "Nudity experiment finished. Metrics under: ${RUN_DIR}/metrics"
