#!/bin/bash -l
# ============================================================================
# SLURM Array Job – SD NSFW reduced_dim Ablation Study (5 dims × 3 seeds = 15)
# ============================================================================
# Varies: reduced_dim in {8, 16, 32, 64, 128}, seed in {42, 1, 2}
# Fixed:  lambda=10.0, lr=5e-6, epochs=5, Adam, base_method=nsfw,
#         targets=[attn2.to_q, attn2.to_k, attn2.to_v]
#
# Outputs per job: metrics JSON via --metrics-out
#
# HOW TO USE:
#   cd SD
#   sbatch scripts/slurm_sd_nsfw_reduced_dim_ablation.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-abl-dim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-14

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
source "$SCRATCH/sd_venv/bin/activate"
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl/taming-transformers:$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Hugging Face token
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

# Cache
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
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

RESULTS_BASE="${SCRATCH:-$HOME}/intact/SD/ablation"
mkdir -p "$RESULTS_BASE"

echo "Starting reduced_dim ablation on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

# ============================================================================
# Grid: 5 reduced_dims × 3 seeds = 15 jobs
# Mapping: IDX = dim_index * 3 + seed_index
# ============================================================================
REDUCED_DIMS=(8 16 32 64 128)
SEEDS=(42 1 2)

IDX=${SLURM_ARRAY_TASK_ID}

DIM_IDX=$(( IDX / 3 ))
SEED_IDX=$(( IDX % 3 ))

DIM=${REDUCED_DIMS[$DIM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "============================================"
echo "SD NSFW Ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  reduced_dim=${DIM}  seed=${SEED}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/sd_nsfw_abl_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"
METRICS_OUT="${RESULTS_BASE}/reduced_dim_${DIM}_seed_${SEED}/metrics.json"

mkdir -p "$(dirname "${METRICS_OUT}")"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override varying hyperparams
cfg["intact"]["reduced_dim"]    = int("${DIM}")
cfg["pipeline"]["seed"]         = int("${SEED}")

# Fix remaining hyperparams
cfg["unlearn"]["lr"]     = 5e-6
cfg["unlearn"]["epochs"] = 5
cfg["intact"]["lambda_interval"] = 10.0

# Override paths for Helios
cfg["paths"]["sd_ckpt"]         = "$SCRATCH/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
cfg["paths"]["output_dir"]      = "${RESULTS_BASE}/reduced_dim_${DIM}_seed_${SEED}"
cfg["paths"]["model_save_dir"]  = "${RESULTS_BASE}/reduced_dim_${DIM}_seed_${SEED}/models"
cfg["paths"]["logs_dir"]        = "${RESULTS_BASE}/reduced_dim_${DIM}_seed_${SEED}/logs"
cfg["paths"]["nsfw_data"]       = "$SCRATCH/data/nsfw"
cfg["paths"]["not_nsfw_data"]   = "$SCRATCH/data/not-nsfw"

# Tag the wandb run
cfg["wandb"]["tags"].append("reduced_dim_ablation")
cfg["wandb"]["group"] = "nsfw-abl-reduced-dim"

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py \
    --config "${TMPCONFIG}" \
    --metrics-out "${METRICS_OUT}"

echo "SD NSFW Ablation – Job ${IDX} (dim=${DIM}, seed=${SEED}) complete."