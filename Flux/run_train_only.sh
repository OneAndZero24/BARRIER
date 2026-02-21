#!/usr/bin/env bash
# =============================================================================
# Run Flux InTAct Standalone Training (no eval pipeline)
# =============================================================================
# For quick training without the full eval/wandb pipeline.
# Reads same YAML config but only runs intact_train.py.
#
# Usage:
#   cd Flux
#   bash run_train_only.sh configs/intact/pipeline_concept.yaml
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"
export WANDB_CACHE_DIR="/shared/results/common/miksa/.cache/wandb"
export CLIP_CACHE_DIR="/shared/results/common/miksa/.cache/clip"

CONFIG="${1:-configs/intact/pipeline_concept.yaml}"

echo "============================================"
echo " Flux InTAct — Training Only"
echo " Config: $CONFIG"
echo "============================================"

python intact_train.py --config "$CONFIG"
