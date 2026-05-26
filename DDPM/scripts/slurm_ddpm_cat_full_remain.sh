#!/bin/bash
# ============================================================================
# SLURM Job – DDPM CIFAR-10 Cat Forgetting with Full Remain Set
# ============================================================================
# Goal:
#   Unlearn a single selected class (cat) using the requested InTAct setup,
#   then generate the cat-focused visualization grid from the resulting run.
#
# Requested settings:
#   target class = cat (label 3)
#   k = 32
#   lambda = 5
#   lr = 10e-4
#   steps = 3k
#   targets = ["attn.0.q", "attn.0.k", "attn.0.v", "attn_1.q", "attn_1.k",
#              "attn_1.v", "attn.1.q", "attn.1.k", "attn.1.v",
#              "cemb.dense.0", "cemb.dense.1"]
#   remain fraction = 1.0 (full remain, not a fraction sweep)
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_cat_full_remain.sh
# ============================================================================

#SBATCH --job-name=ddpm-cat-full
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100

set -euo pipefail

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH="/home/miksa/InTAct-Unl:${PYTHONPATH:-}"

# ---- Fixed setup ----
FORGET_CLASS=3
LR=1e-3
NITERS=3000
LAMBDA=5.0
REDUCED_DIM=32
METHOD="rl"
USE_ACTUAL_BOUNDS=true
REMAIN_FRAC=1.0
BASE_SEED=1234
DEFAULT_PRETRAIN_ROOT="/shared/results/common/miksa/intact/DDPM/results/cifar10"
LOCAL_PRETRAIN_ROOT="$PWD/results/cifar10"
TRAIN_BASE_IF_MISSING="${TRAIN_BASE_IF_MISSING:-true}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-cifar10_train.yml}"

# Optional override, e.g.:
#   sbatch --export=ALL,PRETRAINED_CKPT_FOLDER=/path/to/base_run scripts/slurm_ddpm_cat_full_remain.sh
PRETRAINED_CKPT_FOLDER="${PRETRAINED_CKPT_FOLDER:-}"

# Base reference dataset for FID/classifier evaluation.
REF_DATASET_DIR="$PWD/cifar10_without_label_${FORGET_CLASS}"
CHECKPOINT_ROOT="/shared/results/common/miksa/intact/DDPM/results/fulleval/cat_full_remain"
OUTPUT_ROOT="/shared/results/common/miksa/intact/DDPM/results/fulleval/cat_full_remain"

echo "============================================"
echo "DDPM Cat Full-Remain Run"
echo "  forget_class=${FORGET_CLASS}  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "  reduced_dim=${REDUCED_DIM}  method=${METHOD}  use_actual_bounds=${USE_ACTUAL_BOUNDS}"
echo "  remain_fraction=${REMAIN_FRAC}  seed=${BASE_SEED}"
echo "============================================"

if [[ -z "${PRETRAINED_CKPT_FOLDER}" ]]; then
    LATEST_CKPT_FILE="$(find "${DEFAULT_PRETRAIN_ROOT}" -type f -path '*/ckpts/ckpt.pth' 2>/dev/null | sort | tail -n1 || true)"
    if [[ -n "${LATEST_CKPT_FILE}" ]]; then
        PRETRAINED_CKPT_FOLDER="$(dirname "$(dirname "${LATEST_CKPT_FILE}")")"
    fi
fi

if [[ -z "${PRETRAINED_CKPT_FOLDER}" ]]; then
    LATEST_LOCAL_CKPT_FILE="$(find "${LOCAL_PRETRAIN_ROOT}" -type f -path '*/ckpts/ckpt.pth' 2>/dev/null | sort | tail -n1 || true)"
    if [[ -n "${LATEST_LOCAL_CKPT_FILE}" ]]; then
        PRETRAINED_CKPT_FOLDER="$(dirname "$(dirname "${LATEST_LOCAL_CKPT_FILE}")")"
    fi
fi

if [[ -z "${PRETRAINED_CKPT_FOLDER}" || ! -f "${PRETRAINED_CKPT_FOLDER}/ckpts/ckpt.pth" ]]; then
    if [[ "${TRAIN_BASE_IF_MISSING}" == "true" ]]; then
        echo "No valid pretrained checkpoint found. Training base DDPM first."
        python train.py --config "${BASE_TRAIN_CONFIG}" --mode train --seed "${BASE_SEED}"
        LATEST_LOCAL_CKPT_FILE="$(find "${LOCAL_PRETRAIN_ROOT}" -type f -path '*/ckpts/ckpt.pth' 2>/dev/null | sort | tail -n1 || true)"
        if [[ -n "${LATEST_LOCAL_CKPT_FILE}" ]]; then
            PRETRAINED_CKPT_FOLDER="$(dirname "$(dirname "${LATEST_LOCAL_CKPT_FILE}")")"
        fi
    fi
fi

if [[ -z "${PRETRAINED_CKPT_FOLDER}" || ! -f "${PRETRAINED_CKPT_FOLDER}/ckpts/ckpt.pth" ]]; then
    echo "Error: could not locate a valid pretrained DDPM checkpoint folder." >&2
    echo "Set TRAIN_BASE_IF_MISSING=true to auto-train, or provide PRETRAINED_CKPT_FOLDER." >&2
    echo "Set PRETRAINED_CKPT_FOLDER to a base run directory containing ckpts/ckpt.pth" >&2
    echo "Example: PRETRAINED_CKPT_FOLDER=/shared/results/common/miksa/intact/DDPM/results/cifar10/<run_ts> sbatch scripts/slurm_ddpm_cat_full_remain.sh" >&2
    exit 1
fi

echo "Using pretrained checkpoint folder: ${PRETRAINED_CKPT_FOLDER}"

if [[ ! -d "${REF_DATASET_DIR}" ]]; then
  echo "Reference dataset not found at ${REF_DATASET_DIR}; generating it now."
  python save_base_dataset.py \
    --dataset cifar10 \
    --label_to_forget "${FORGET_CLASS}" \
    --data_path ../data
fi

TMPCONFIG="/tmp/ddpm_cat_full_remain_${SLURM_JOB_ID:-manual}.yaml"

python - <<PYEOF
import os
import yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

forget_class = int("${FORGET_CLASS}")
lr = float("${LR}")
niters = int("${NITERS}")
lam = float("${LAMBDA}")
reduced_dim = int("${REDUCED_DIM}")
method = "${METHOD}"
use_actual_bounds = "${USE_ACTUAL_BOUNDS}".lower() == "true"
remain_frac = float("${REMAIN_FRAC}")
seed = int("${BASE_SEED}")

cfg["paths"]["ref_dataset_dir"] = os.path.abspath("${REF_DATASET_DIR}")
cfg["paths"]["pretrained_ckpt_folder"] = os.path.abspath("${PRETRAINED_CKPT_FOLDER}")
cfg["unlearn"]["label_to_forget"] = forget_class
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["n_iters"] = niters
cfg["unlearn"]["method"] = method
cfg.setdefault("pipeline", {})
cfg["pipeline"]["seed"] = seed

cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = lam
cfg["intact"]["reduced_dim"] = reduced_dim
cfg["intact"]["use_actual_bounds"] = use_actual_bounds
cfg["intact"]["remain_fraction"] = remain_frac
cfg["intact"]["remain_subset_seed"] = seed
cfg["intact"]["targets"] = [
    "attn.0.q",
    "attn.0.k",
    "attn.0.v",
    "attn_1.q",
    "attn_1.k",
    "attn_1.v",
    "attn.1.q",
    "attn.1.k",
    "attn.1.v",
    "cemb.dense.0",
    "cemb.dense.1",
]

cfg.setdefault("evaluate", {}).setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

cfg["wandb"]["group"] = "cifar10-cat-full-remain"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "cat",
    "full-remain",
    "single-class",
    "k_32",
    "lambda_5",
    "lr_1e-3",
    "steps_3000",
]

suffix = (
    f"cat_fullremain_fc{forget_class}"
    f"_seed{seed}_lr{lr}_ni{niters}_lam{lam}_k{reduced_dim}"
)
cfg["paths"]["output_dir"] = os.path.join("${OUTPUT_ROOT}", suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join("${CHECKPOINT_ROOT}", suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

RUN_CKPT_ROOT="${CHECKPOINT_ROOT}/cat_fullremain_fc${FORGET_CLASS}_seed${BASE_SEED}_lr${LR}_ni${NITERS}_lam${LAMBDA}_k${REDUCED_DIM}"
RUN_CKPT_DIR="$(find "${RUN_CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n1 || true)"

if [[ -z "${RUN_CKPT_DIR}" ]]; then
  echo "Warning: could not locate the timestamped checkpoint directory under ${RUN_CKPT_ROOT}." >&2
  echo "Skipping visualization step." >&2
  exit 0
fi

echo "Rendering visualization from ${RUN_CKPT_DIR}"
CKPT_FOLDER="${RUN_CKPT_DIR}" bash scripts/render_ddpm_fig3_cat_grid.sh

echo "Cat full-remain run complete."