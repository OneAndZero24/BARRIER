#!/bin/bash
# ============================================================================
# SLURM – STEREO Nudity Generate + Score
# ============================================================================
# Generates images from the selected nudity prompts with the unlearned model
# and scores the outputs with NudeNet at threshold 0.2.
# ============================================================================

#SBATCH --job-name=stereo-nudity-generate-score
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm

REPO_ROOT="$HOME/InTAct-Unl/SD"

cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(cd .. && pwd)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STEREO_ROOT="$REPO_ROOT/stereo"
PROMPTS_CSV="$REPO_ROOT/prompts/Nudity_eta_3_K_16.csv"
RUN_ROOT="${STEREO_ROOT}/runs"
RESULTS_ROOT="${STEREO_ROOT}/results"

MODEL_NAME="compvis-intact-nsfw-targets_tgth_675706798c_n3-lambda_0.5-lr_5e-06"
MODEL_DIR="/shared/results/common/miksa/intact/SD/models"
BASE_MODEL_ID="CompVis/stable-diffusion-v1-4"

SAVE_PATH="${RUN_ROOT}/nudity_check"
RESULTS_CSV="${RESULTS_ROOT}/nudity_check_results.csv"

mkdir -p "${RUN_ROOT}" "${RESULTS_ROOT}" "${SAVE_PATH}"

python scripts/stereo_nudity_generate_and_score.py \
  --prompts-path "${PROMPTS_CSV}" \
  --model-name "${MODEL_NAME}" \
  --model-dir "${MODEL_DIR}" \
  --save-path "${SAVE_PATH}" \
  --base-model-path "${BASE_MODEL_ID}" \
  --threshold 0.2 \
  --num-samples 1 \
  --from-case 0 \
  --results-csv "${RESULTS_CSV}"