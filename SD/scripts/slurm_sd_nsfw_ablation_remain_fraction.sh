#!/bin/bash -l
# ============================================================================
# SLURM Array Job – SD NSFW Remain-Set Bounds Fraction Ablation
# ============================================================================
# Varies: bounds_remain_fraction in {0.10, 0.25, 0.50, 0.75, 1.0}
#         seed in {42, 1, 2}
# Fixed:  bounds_forget_fraction = 1.0
#         lambda=0.5, lr=5e-6, epochs=3, Adam, base_method=nsfw,
#         reduced_dim=64, targets=[attn2.to_q, attn2.to_k, attn2.to_v]
#         use_actual_bounds=true, transf. block 2 (cross-attn QKV)
#
# Total: 5 fractions x 3 seeds = 15 jobs
#
# HOW TO USE:
#   cd SD
#   sbatch scripts/slurm_sd_nsfw_ablation_remain_fraction.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-abl-rf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-14

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
source "$SCRATCH/sd_venv/bin/activate"
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl/taming-transformers:$HOME/InTAct-Unl:${PYTHONPATH:-}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb_data"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_DATA_DIR"

RESULTS_BASE="${SCRATCH:-$HOME}/intact/SD/ablation"
mkdir -p "$RESULTS_BASE"

cleanup() {
    local rc=$?
    echo "=== CLEANUP (exit code $rc) ==="
    rm -f "${TMPCONFIG:-/dev/null}" 2>/dev/null || true
    if [ -n "${PERJOB_DIR:-}" ] && [ -d "${PERJOB_DIR}" ]; then
        find "${PERJOB_DIR}" -mindepth 1 -maxdepth 1 -not -name 'metrics.json' \
            -exec rm -rf {} + 2>/dev/null || true
        echo "Cleaned per-job dir: ${PERJOB_DIR}"
    fi
    exit $rc
}
trap cleanup EXIT

echo "Starting remain-fraction ablation on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

# ============================================================================
# Grid: 5 fractions x 3 seeds = 15 jobs
# Mapping: IDX = frac_index * 3 + seed_index
# ============================================================================
FRACTIONS=(0.10 0.25 0.50 0.75 1.0)
SEEDS=(42 1 2)

IDX=${SLURM_ARRAY_TASK_ID}

FRAC_IDX=$(( IDX / 3 ))
SEED_IDX=$(( IDX % 3 ))

FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

FRAC_PCT=$(printf "%.0f" "$(echo "${FRACTION} * 100" | bc)")

echo "============================================"
echo "SD NSFW Remain-Fraction Ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  bounds_remain_fraction=${FRACTION} (${FRAC_PCT}%)  seed=${SEED}"
echo "============================================"

TMPCONFIG="/tmp/sd_nsfw_abl_rf_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"
METRICS_OUT="${RESULTS_BASE}/remain_frac_${FRAC_PCT}_seed_${SEED}/metrics.json"
PERJOB_DIR="$(dirname "${METRICS_OUT}")"

mkdir -p "$PERJOB_DIR"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["intact"]["bounds_forget_fraction"]  = 1.0
cfg["intact"]["bounds_remain_fraction"]  = float("${FRACTION}")
cfg["pipeline"]["seed"]                  = int("${SEED}")

cfg["unlearn"]["lr"]     = 5e-6
cfg["unlearn"]["epochs"] = 3
cfg["intact"]["lambda_interval"] = 0.5
cfg["intact"]["reduced_dim"]     = 64
cfg["intact"]["use_actual_bounds"] = True

# Target all cross-attention QKV layers (matches known-working combo19 config)
# No block restriction - targets are already correct from the base config

suffix = f"remain_frac_${FRAC_PCT}_seed_${SEED}"
cfg["paths"]["sd_ckpt"]         = "$SCRATCH/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
cfg["paths"]["output_dir"]      = os.path.join("${RESULTS_BASE}", suffix)
cfg["paths"]["model_save_dir"]  = os.path.join("${RESULTS_BASE}", suffix, "models")
cfg["paths"]["logs_dir"]        = os.path.join("${RESULTS_BASE}", suffix, "logs")
cfg["paths"]["nsfw_data"]       = "$SCRATCH/data/nsfw"
cfg["paths"]["not_nsfw_data"]   = "$SCRATCH/data/not-nsfw"

cfg["wandb"]["tags"].append("remain_fraction_ablation")
cfg["wandb"]["group"] = "nsfw-abl-remain-fraction"

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py \
    --config "${TMPCONFIG}" \
    --metrics-out "${METRICS_OUT}"

echo "SD NSFW Remain-Fraction Ablation – Job ${IDX} (frac=${FRAC_PCT}%, seed=${SEED}) complete."