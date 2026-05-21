#!/bin/bash
# ============================================================================
# DDPM Figure 3 Render – CIFAR-10 Cat Forgetting Grid
# ============================================================================
# Generates the paper-style visualization for cat forgetting:
#   5 samples from the forgotten class (cat)
#   1 sample from each remaining CIFAR-10 class
#
# The script expects a trained/forgotten checkpoint folder and saves the grid
# as sample-fig3_cat_grid.png inside that folder.
# ============================================================================

set -euo pipefail

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ---- Input checkpoint folder ----
CKPT_FOLDER="${CKPT_FOLDER:-}"
if [[ -z "${CKPT_FOLDER}" ]]; then
  echo "Error: set CKPT_FOLDER to the DDPM run directory to visualize." >&2
  echo "Example: CKPT_FOLDER=results/fulleval/<run>/ python ..." >&2
  exit 1
fi

COND_SCALE="${COND_SCALE:-2.0}"

python sample.py \
  --config cifar10_sample.yml \
  --ckpt_folder "${CKPT_FOLDER}" \
  --mode visualization \
  --cond_scale "${COND_SCALE}" \
  --visualization_name fig3_cat_grid \
  --classes_to_generate "3,0,1,2,4,5,6,7,8,9" \
  --visualization_counts "5,1,1,1,1,1,1,1,1,1"

echo "Figure 3 grid saved to ${CKPT_FOLDER}/sample-fig3_cat_grid.png"