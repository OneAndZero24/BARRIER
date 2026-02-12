#!/bin/bash
# ============================================================================
# Stable Diffusion Sweep Runner
# ============================================================================
# Usage:
#   ./run_sweep.sh sweep_class
#   ./run_sweep.sh sweep_nsfw

set -e

MAIN_DIR="configs"
PROJECT_NAME="intact-sd"

run_sweep_and_agent() {
  SWEEP_NAME="$1"
  
  YAML_PATH="$MAIN_DIR/${SWEEP_NAME}.yaml"
  
  if [ ! -f "$YAML_PATH" ]; then
    echo "Error: YAML file '${SWEEP_NAME}.yaml' not found in $MAIN_DIR"
    exit 1
  fi
  
  echo "Running wandb sweep for: $SWEEP_NAME"
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" > ${SWEEP_NAME}_temp_output.txt 2>&1
  
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' ${SWEEP_NAME}_temp_output.txt)
  
  rm ${SWEEP_NAME}_temp_output.txt

  echo "Starting WandB agent for sweep ID: $SWEEP_ID"
  wandb agent "$SWEEP_ID"
}

# Main execution
if [ $# -eq 0 ]; then
  echo "Usage: $0 <sweep_name>"
  echo "Available sweeps:"
  echo "  - sweep_class"
  echo "  - sweep_nsfw"
  exit 1
fi

SWEEP_NAME="$1"
run_sweep_and_agent "$SWEEP_NAME"
