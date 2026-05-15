#!/bin/bash -l
# ============================================================================
# SLURM Job – NSFW BIG Sweep (Flux) — Single GPU
# ============================================================================
# Runs a dedicated wandb sweep for NSFW erasure on Flux with H100 / 256 GB
# nodes using `configs/intact/sweep_nsfw_big.yaml`.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_sweep_nsfw_big.sh
#   sbatch scripts/slurm_sweep_nsfw_big.sh <existing_sweep_id>
#
# Resume behavior:
#   - By default, each sbatch submission uses its own sweep namespace,
#     so array tasks attach only to the current submission's sweep.
#   - To resume an older/shared sweep, either pass an explicit sweep ID,
#     or set SWEEP_NAMESPACE=shared.
#   - Set FORCE_NEW_SWEEP=1 to force creation of a fresh sweep in namespace.
# ============================================================================

#SBATCH --job-name=flux-nsfw-big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-4

set -euo pipefail

# ---- Environment ----
ml ML-bundle/24.06a
source "$HOME/venv/bin/activate"
cd "$HOME/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

# Load Hugging Face credentials from the saved token file if present.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/net/home/plgrid/plgmiksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: no Hugging Face token found; expected $HF_TOKEN_FILE or HF_TOKEN/HUGGINGFACE_HUB_TOKEN"
    exit 1
fi

# wandb configuration (must have permission to create sweeps here)
export WANDB_ENTITY="oneandzero24"
PROJECT_NAME="intact-flux"

# caches (prefer new SCRATCH setup, then legacy, then home fallback)
if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
elif [ -w "/shared/results/common/miksa/intact/SD" ] || [ -w "/shared/results/common/miksa/intact/SD/.cache" ]; then
    CACHE_BASE="/shared/results/common/miksa/intact/SD/.cache"
else
    CACHE_BASE="$HOME/.cache/intact"
fi
export CACHE_ROOT="$CACHE_BASE"

# Keep all tool caches consistent with selected CACHE_ROOT.
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

echo "Starting Flux NSFW BIG sweep on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-none}"

echo "=== launching wandb sweep ==="
SWEEP_NAME="sweep_nsfw_big"
YAML_PATH="configs/intact/${SWEEP_NAME}.yaml"
ARRAY_KEY="${SLURM_ARRAY_JOB_ID:-manual}"

# Default namespace is submission-scoped to avoid mixing with old sweeps.
DEFAULT_SWEEP_NAMESPACE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SWEEP_NAMESPACE="${SWEEP_NAMESPACE:-$DEFAULT_SWEEP_NAMESPACE}"
# IMPORTANT: keep sweep coordination state on a shared filesystem.
# CACHE_ROOT can point to task-local scratch on some clusters, which causes
# each array task to create its own sweep instead of reusing one sweep ID.
SWEEP_STATE_BASE="${SWEEP_STATE_BASE:-$HOME/.cache/intact/wandb-sweeps}"
SWEEP_STATE_DIR="$SWEEP_STATE_BASE/${PROJECT_NAME}/${SWEEP_NAME}_${SWEEP_NAMESPACE}"
SWEEP_ID_FILE="$SWEEP_STATE_DIR/sweep.id"
SWEEP_LOCK_DIR="$SWEEP_STATE_DIR/create.lock"

# Legacy per-array location (kept for backward-compatible migration).
LEGACY_SWEEP_STATE_DIR="$CACHE_ROOT/wandb/${SWEEP_NAME}_${ARRAY_KEY}"
LEGACY_SWEEP_ID_FILE="$LEGACY_SWEEP_STATE_DIR/sweep.id"

SWEEP_WAIT_SECONDS="${SWEEP_WAIT_SECONDS:-1800}"
SWEEP_POLL_SECONDS="${SWEEP_POLL_SECONDS:-5}"
FORCE_NEW_SWEEP="${FORCE_NEW_SWEEP:-0}"

mkdir -p "$SWEEP_STATE_DIR"
echo "Sweep coordination directory: $SWEEP_STATE_DIR"
echo "Sweep namespace: $SWEEP_NAMESPACE"

create_sweep() {
    local out
    local id

    echo "Creating new sweep from $YAML_PATH"
    set -x
    # Capture both stdout and stderr because wandb may print sweep info to stderr.
    out=$(wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" 2>&1)
    set +x
    echo "$out"

    id=$(echo "$out" | awk '/wandb agent/{ match($0, /wandb agent ([^ ]+)/, arr); print arr[1]; }')
    if [ -z "$id" ]; then
        id=$(echo "$out" | awk '/Creating sweep with ID/{ match($0, /ID: ([^ ]+)/, arr); print arr[1]; }')
    fi
    if [ -z "$id" ]; then
        echo "ERROR: Failed to parse sweep ID from output"
        return 1
    fi

    echo "$id" > "$SWEEP_ID_FILE"
    echo "Saved sweep ID to $SWEEP_ID_FILE"
}

SWEEP_ID=""

if [ -n "${1:-}" ]; then
    SWEEP_ID="$1"
    echo "Using provided sweep ID: $SWEEP_ID"
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
    echo "Saved provided sweep ID to $SWEEP_ID_FILE"
elif [ "$FORCE_NEW_SWEEP" = "1" ]; then
    echo "FORCE_NEW_SWEEP=1: creating a fresh sweep"
    if mkdir "$SWEEP_LOCK_DIR" 2>/dev/null; then
        cleanup_lock() {
            rmdir "$SWEEP_LOCK_DIR" 2>/dev/null || true
        }
        trap cleanup_lock EXIT
        create_sweep
        if [ -s "$SWEEP_ID_FILE" ]; then
            SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
        fi
        trap - EXIT
        cleanup_lock
    else
        echo "Another task is creating a sweep. Waiting for $SWEEP_ID_FILE"
        MAX_POLLS=$((SWEEP_WAIT_SECONDS / SWEEP_POLL_SECONDS))
        if [ "$MAX_POLLS" -lt 1 ]; then
            MAX_POLLS=1
        fi
        for _ in $(seq 1 "$MAX_POLLS"); do
            if [ -s "$SWEEP_ID_FILE" ]; then
                break
            fi
            sleep "$SWEEP_POLL_SECONDS"
        done
        if [ -s "$SWEEP_ID_FILE" ]; then
            SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
        fi
    fi
    if [ -z "${SWEEP_ID:-}" ]; then
        echo "ERROR: FORCE_NEW_SWEEP requested but sweep ID is empty; expected file: $SWEEP_ID_FILE"
        exit 1
    fi
elif [ -s "$SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
    echo "Reusing existing sweep ID (shared state): $SWEEP_ID"
elif [ -s "$LEGACY_SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$LEGACY_SWEEP_ID_FILE")"
    echo "Reusing legacy per-array sweep ID: $SWEEP_ID"
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
    echo "Migrated legacy sweep ID to shared state: $SWEEP_ID_FILE"
else
    # Lock-based leader election: first task that creates the lock directory creates the sweep.
    if mkdir "$SWEEP_LOCK_DIR" 2>/dev/null; then
        cleanup_lock() {
            rmdir "$SWEEP_LOCK_DIR" 2>/dev/null || true
        }
        trap cleanup_lock EXIT
        create_sweep
        trap - EXIT
        cleanup_lock
    else
        echo "Another array task is creating the sweep. Waiting for $SWEEP_ID_FILE"
        MAX_POLLS=$((SWEEP_WAIT_SECONDS / SWEEP_POLL_SECONDS))
        if [ "$MAX_POLLS" -lt 1 ]; then
            MAX_POLLS=1
        fi

        for _ in $(seq 1 "$MAX_POLLS"); do
            if [ -s "$SWEEP_ID_FILE" ]; then
                break
            fi
            if [ ! -d "$SWEEP_LOCK_DIR" ] && [ ! -s "$SWEEP_ID_FILE" ]; then
                echo "Sweep lock disappeared and no ID file found; retrying lock acquisition"
                if mkdir "$SWEEP_LOCK_DIR" 2>/dev/null; then
                    cleanup_lock() {
                        rmdir "$SWEEP_LOCK_DIR" 2>/dev/null || true
                    }
                    trap cleanup_lock EXIT
                    create_sweep
                    trap - EXIT
                    cleanup_lock
                    break
                fi
            fi
            sleep "$SWEEP_POLL_SECONDS"
        done
    fi

    if [ -s "$SWEEP_ID_FILE" ]; then
        SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
    fi

    if [ -z "${SWEEP_ID:-}" ]; then
        echo "ERROR: sweep ID is empty; expected file: $SWEEP_ID_FILE"
        exit 1
    fi
fi

echo "Starting WandB agent for sweep ID: $SWEEP_ID"
wandb agent "$SWEEP_ID"

# wandb agent blocks until the sweep completes or is stopped

echo "Flux NSFW BIG sweep finished."
