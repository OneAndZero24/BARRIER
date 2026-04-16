#!/bin/bash -l
# ============================================================================
# SLURM Job – NSFW Sweep (Flux) — Single GPU
# ============================================================================
# Runs a full wandb sweep for NSFW erasure on Flux with H100 / 256 GB nodes.
# The sweep configuration lives at `configs/intact/sweep_nsfw.yaml`.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_sweep_nsfw.sh
#   sbatch scripts/slurm_sweep_nsfw.sh <existing_sweep_id>
# ============================================================================

#SBATCH --job-name=flux-nsfw-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
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
export WANDB_ENTITY="oneandzero24"    # adjust to your account/org
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
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"

echo "Starting Flux NSFW sweep on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-none}"

echo "=== launching wandb sweep ==="
SWEEP_NAME="sweep_nsfw"
YAML_PATH="configs/intact/${SWEEP_NAME}.yaml"
ARRAY_KEY="${SLURM_ARRAY_JOB_ID:-manual}"
SWEEP_STATE_DIR="$CACHE_ROOT/wandb/${SWEEP_NAME}_${ARRAY_KEY}"
SWEEP_ID_FILE="$SWEEP_STATE_DIR/sweep.id"
SWEEP_LOCK_DIR="$SWEEP_STATE_DIR/create.lock"
SWEEP_WAIT_SECONDS="${SWEEP_WAIT_SECONDS:-1800}"
SWEEP_POLL_SECONDS="${SWEEP_POLL_SECONDS:-5}"

mkdir -p "$SWEEP_STATE_DIR"

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

if [ -n "${1:-}" ]; then
    SWEEP_ID="$1"
    echo "Using provided sweep ID: $SWEEP_ID"
elif [ -s "$SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$SWEEP_ID_FILE")"
    echo "Reusing existing sweep ID for this array job: $SWEEP_ID"
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

echo "Flux NSFW sweep finished."
