#!/usr/bin/env bash
# =============================================================================
# Run Flux InTAct Pipeline (Concept Erasure)
# =============================================================================
# Usage:
#   cd Flux
#   bash run_intact.sh [--config path/to/config.yaml] [--no-wandb]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/intact/pipeline_concept.yaml}"
shift 2>/dev/null || true

echo "============================================"
echo " Flux InTAct Pipeline"
echo " Config: $CONFIG"
echo "============================================"

python intact_pipeline.py --config "$CONFIG" "$@"
