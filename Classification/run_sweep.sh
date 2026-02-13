#!/bin/bash
# ============================================================================
# Classification Sweep Runner
# ============================================================================
# Usage:
#   ./run_sweep.sh sweep_classwise
#   ./run_sweep.sh sweep_random

set -e

MAIN_DIR="configs"
PROJECT_NAME="intact-classification"

run_sweep_and_agent() {
  SWEEP_NAME="$1"
  
  YAML_PATH="$MAIN_DIR/${SWEEP_NAME}.yaml"
  
  if [ ! -f "$YAML_PATH" ]; then
    echo "Error: YAML file '${SWEEP_NAME}.yaml' not found in $MAIN_DIR"
    exit 1
  fi
  
  echo "Running wandb sweep for: $SWEEP_NAME"
  SWEEP_ID=$(wandb sweep "$YAML_PATH" 2>&1 | grep "wandb agent" | awk '{print $NF}')
  
  echo "Starting WandB agent for sweep ID: $SWEEP_ID"
  wandb agent "$SWEEP_ID"
}

# Main execution
if [ $# -eq 0 ]; then
  echo "Usage: $0 <sweep_name>"
  echo "Available sweeps:"
  echo "  - sweep_classwise"
  echo "  - sweep_random"
  exit 1
fi

SWEEP_NAME="$1"
run_sweep_and_agent "$SWEEP_NAME"
