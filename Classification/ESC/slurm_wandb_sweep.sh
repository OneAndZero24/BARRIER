#!/bin/bash -l
# ============================================================================
# SLURM + wandb sweep — ESC vs InTAct Tiny-ImageNet ViT (cookbook pattern)
#
# Array job with lock-based sweep creation: the first task that wins a mkdir
# lock creates the wandb sweep; all tasks join it and run `wandb agent`.
# Sweep state lives on a SHARED filesystem so every task joins the same sweep.
#
# Usage:
#   sbatch slurm_wandb_sweep.sh                                  # new sweep
#   SWEEP_YAML=configs/sweep_bayes_intact_tinyimagenet.yaml \
#     SWEEP_NAME=bayes-40 SWEEP_RUNS=40 sbatch slurm_wandb_sweep.sh
#   sbatch slurm_wandb_sweep.sh <existing_sweep_id>              # resume
# ============================================================================
#SBATCH --job-name=esc-intact-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=big
#SBATCH --partition=dgxa100
#SBATCH --array=0-3

set -euo pipefail

# ---- environment (edit here or export on submit) ----
ESC_DIR="${ESC_DIR:-$HOME/BARRIER/Classification/ESC}"
CONDA_ENV="${CONDA_ENV:-ESC}"
CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"

# ---- sweep config ----
SWEEP_YAML="${SWEEP_YAML:-configs/sweep_bayes_intact_tinyimagenet.yaml}"
SWEEP_NAME="${SWEEP_NAME:-intact-bayes-40}"
PROJECT_NAME="${WANDB_PROJECT:-esc-intact-tinyimagenet}"
ENTITY="${WANDB_ENTITY:-oneandzero24}"
SWEEP_RUNS="${SWEEP_RUNS:-40}"                 # total runs across all agents

# ---- caches (never in $HOME) ----
CACHE_BASE="${CACHE_BASE:-/shared/results/common/miksa/.cache}"
export HF_HOME="$CACHE_BASE/huggingface" TORCH_HOME="$CACHE_BASE/torch" XDG_CACHE_HOME="$CACHE_BASE"
export WANDB_DIR="$CACHE_BASE/wandb" WANDB_CACHE_DIR="$CACHE_BASE/wandb"
export TMPDIR="$CACHE_BASE/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

# ---- conda ----
[ -f "$CONDA_SH" ] || { echo "conda.sh not found at $CONDA_SH"; exit 1; }
# shellcheck disable=SC1091
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ESC_DIR"

# ---- sweep coordination (shared filesystem!) ----
SWEEP_STATE_DIR="${SWEEP_STATE_DIR:-/shared/results/common/miksa/esc-intact/sweeps/${SWEEP_NAME}_${SLURM_ARRAY_JOB_ID}}"
SWEEP_ID_FILE="$SWEEP_STATE_DIR/sweep.id"
SWEEP_LOCK_DIR="$SWEEP_STATE_DIR/create.lock"
SWEEP_WAIT_SECONDS="${SWEEP_WAIT_SECONDS:-1800}"
SWEEP_POLL_SECONDS="${SWEEP_POLL_SECONDS:-5}"
mkdir -p "$SWEEP_STATE_DIR"

create_sweep() {
    local out id
    out=$(wandb sweep --project "$PROJECT_NAME" --entity "$ENTITY" --name "$SWEEP_NAME" "$SWEEP_YAML" 2>&1)
    echo "$out"
    id=$(echo "$out" | awk '/wandb agent/{ match($0, /wandb agent ([^ ]+)/, arr); print arr[1]; }')
    [ -z "$id" ] && id=$(echo "$out" | awk '/Creating sweep with ID/{ match($0, /ID: ([^ ]+)/, arr); print arr[1]; }')
    [ -z "$id" ] && { echo "ERROR: failed to parse sweep ID"; return 1; }
    echo "$id" > "$SWEEP_ID_FILE"
}

SWEEP_ID=""
if [ -n "${1:-}" ]; then
    SWEEP_ID="$1"; echo "$SWEEP_ID" > "$SWEEP_ID_FILE"   # resume existing sweep
elif [ -s "$SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$SWEEP_ID_FILE")"                   # shared state already exists
else
    if mkdir "$SWEEP_LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$SWEEP_LOCK_DIR" 2>/dev/null || true' EXIT
        create_sweep
    else
        echo "Another task is creating the sweep; waiting for $SWEEP_ID_FILE"
        for _ in $(seq 1 $(( SWEEP_WAIT_SECONDS / SWEEP_POLL_SECONDS ))); do
            [ -s "$SWEEP_ID_FILE" ] && break
            sleep "$SWEEP_POLL_SECONDS"
        done
    fi
    [ -s "$SWEEP_ID_FILE" ] && SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
fi

[ -z "${SWEEP_ID:-}" ] && { echo "ERROR: sweep ID is empty"; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID}: agent for sweep $SWEEP_ID"

# distribute total runs evenly across array tasks
SLURM_ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-4}"
COUNT_PER_AGENT=$(( (SWEEP_RUNS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
echo "runs/agent=${COUNT_PER_AGENT}"
wandb agent --count "$COUNT_PER_AGENT" "$SWEEP_ID"