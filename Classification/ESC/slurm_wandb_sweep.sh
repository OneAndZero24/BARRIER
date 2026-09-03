#!/bin/bash -l
# ============================================================================
# SLURM + wandb sweep - ESC vs InTAct Tiny-ImageNet ViT (cookbook pattern)
#
# Fully self-contained: preflights the environment, exports the wandb API key
# from ~/.netrc, and uses mkdir-lock leader election so the first array task
# creates the sweep and the rest join it.  Failures are loud and propagate to
# all tasks (no silent 30-min hangs).
#
# Usage - just submit:
#   sbatch slurm_wandb_sweep.sh
#
# Optional knobs (export before sbatch):
#   SWEEP_YAML=configs/sweep_grid_esc_tinyimagenet.yaml  - ESC baseline grid
#   SWEEP_RUNS=60 SWEEP_NAME=bayes-60                   - more runs / rename
#   WANDB_ENTITY=<user> WANDB_PROJECT=<proj>            - override target
#   CONDA_ENV=/path/to/env ESC_DIR=/path/to/ESC         - override layout
# ============================================================================
#SBATCH --job-name=esc-intact-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=big
#SBATCH --partition=dgxa100
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "===== task $TASK_ID on $(hostname) $(date '+%F %T') ====="

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# SLURM starts jobs in the submission directory, so $PWD is the project dir.
# (BASH_SOURCE[0] is NOT reliable on compute nodes - it points at the spooled
# copy of the script under /var/spool/slurmd/job*.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESC_DIR="${ESC_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-/shared/results/common/miksa/envs/ESC}"
CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"

SWEEP_YAML="${SWEEP_YAML:-configs/sweep_bayes_intact_tinyimagenet.yaml}"
SWEEP_NAME="${SWEEP_NAME:-intact-bayes-40}"
PROJECT_NAME="${WANDB_PROJECT:-esc-intact-tinyimagenet}"
ENTITY="${WANDB_ENTITY:-oneandzero24}"
SWEEP_RUNS="${SWEEP_RUNS:-40}"

CACHE_BASE="${CACHE_BASE:-/shared/results/common/miksa/.cache}"
export HF_HOME="$CACHE_BASE/huggingface" TORCH_HOME="$CACHE_BASE/torch" XDG_CACHE_HOME="$CACHE_BASE"
export WANDB_DIR="$CACHE_BASE/wandb" WANDB_CACHE_DIR="$CACHE_BASE/wandb"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[ -f "$CONDA_SH" ] || { echo "[preflight] conda.sh not found at $CONDA_SH"; exit 1; }
# shellcheck disable=SC1091
source "$CONDA_SH"
conda activate "$CONDA_ENV" || { echo "[preflight] conda activate $CONDA_ENV failed"; exit 1; }
command -v wandb >/dev/null || { echo "[preflight] wandb CLI not in $CONDA_ENV - run: pip install wandb"; exit 1; }

cd "$ESC_DIR" || { echo "[preflight] ESC_DIR not found: $ESC_DIR"; exit 1; }
[ -f "$SWEEP_YAML" ] || { echo "[preflight] $SWEEP_YAML not found in $ESC_DIR"; exit 1; }

# wandb auth: env key > ~/.netrc (login node `wandb login` writes ~/.netrc; /home is shared)
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
    export WANDB_API_KEY="$(awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}' "$HOME/.netrc")"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[preflight] wandb not logged in. Run on the login node:  wandb login"
    exit 1
fi

echo "[preflight] env=$CONDA_ENV dir=$ESC_DIR yaml=$SWEEP_YAML entity=$ENTITY project=$PROJECT_NAME runs=$SWEEP_RUNS"

# ---------------------------------------------------------------------------
# Sweep coordination (shared filesystem!)
# ---------------------------------------------------------------------------
SWEEP_STATE_DIR="${SWEEP_STATE_DIR:-/shared/results/common/miksa/esc-intact/sweeps/${SWEEP_NAME}_${SLURM_ARRAY_JOB_ID}}"
SWEEP_ID_FILE="$SWEEP_STATE_DIR/sweep.id"
SWEEP_LOCK_DIR="$SWEEP_STATE_DIR/create.lock"
SWEEP_FAIL_FILE="$SWEEP_STATE_DIR/create.failed"
SWEEP_WAIT_SECONDS="${SWEEP_WAIT_SECONDS:-1800}"
SWEEP_POLL_SECONDS="${SWEEP_POLL_SECONDS:-5}"
rm -f "$SWEEP_FAIL_FILE"
mkdir -p "$SWEEP_STATE_DIR"

create_sweep() {
    local out id rc
    out=$(wandb sweep --project "$PROJECT_NAME" --entity "$ENTITY" --name "$SWEEP_NAME" "$SWEEP_YAML" 2>&1)
    rc=$?
    echo "$out"                      # always show wandb's output
    [ $rc -eq 0 ] || { echo "ERROR: wandb sweep failed (rc=$rc)"; return 1; }
    id=$(echo "$out" | awk '/wandb agent/{ match($0, /wandb agent ([^ ]+)/, arr); print arr[1]; }')
    [ -z "$id" ] && id=$(echo "$out" | awk '/Creating sweep with ID/{ match($0, /ID: ([^ ]+)/, arr); print arr[1]; }')
    [ -z "$id" ] && { echo "ERROR: failed to parse sweep ID"; return 1; }
    echo "$id" > "$SWEEP_ID_FILE"
}

SWEEP_ID=""
if [ -n "${1:-}" ]; then
    SWEEP_ID="$1"; echo "$SWEEP_ID" > "$SWEEP_ID_FILE"   # attach to existing sweep
elif [ -s "$SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$SWEEP_ID_FILE")"                   # shared state already exists
else
    if mkdir "$SWEEP_LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$SWEEP_LOCK_DIR" 2>/dev/null || true' EXIT
        if create_sweep; then
            echo "[task $TASK_ID] sweep created: $SWEEP_ID_FILE"
        else
            touch "$SWEEP_FAIL_FILE"                     # let waiters abort
            echo "[task $TASK_ID] sweep creation failed"
            exit 1
        fi
    else
        echo "[task $TASK_ID] another task is creating the sweep; waiting..."
        for _ in $(seq 1 $(( SWEEP_WAIT_SECONDS / SWEEP_POLL_SECONDS ))); do
            [ -s "$SWEEP_ID_FILE" ] && break
            if [ -f "$SWEEP_FAIL_FILE" ]; then
                echo "[task $TASK_ID] sweep creation failed on leader; aborting"
                exit 1
            fi
            sleep "$SWEEP_POLL_SECONDS"
        done
    fi
    [ -s "$SWEEP_ID_FILE" ] && SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
fi

[ -z "${SWEEP_ID:-}" ] && { echo "ERROR: sweep ID is empty"; exit 1; }
echo "[task $TASK_ID] agent for sweep $SWEEP_ID"

# distribute total runs evenly across array tasks
SLURM_ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-4}"
COUNT_PER_AGENT=$(( (SWEEP_RUNS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
echo "[task $TASK_ID] runs/agent=${COUNT_PER_AGENT}"
wandb agent --count "$COUNT_PER_AGENT" "$SWEEP_ID"