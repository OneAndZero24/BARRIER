#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search for one Imagenette class
# chain_saw only + QKV on all output blocks
# ============================================================================
#   Each array task runs one (hyperparam combo, class) pair.
#
#   Design goals:
#   - Sweep alpha, lambda, lr, reduced_dim.
#   - Keep alpha to two small values.
#   - Use all output blocks with QKV targets only.
#   - Keep epochs fixed across all jobs.
#   - Use defaults for percentile/infinity settings from config.
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_grid_nonzero_alpha_only.sh
# ============================================================================

#SBATCH --job-name=sd-nonzero-grid
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-23

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- Grid over alpha, lambda, lr, reduced_dim ----
# 24 combos x 1 class = 24 active jobs.
# Objective: maximize UA while keeping FID below 10.
# Prior runs suggest this is most likely in low-lambda / low-lr regimes.
GRID_ALPHAS=(
    0.0 0.7
)
GRID_LAMBDAS=(
    8.0 10.0 12.0
)
GRID_LRS=(
    5e-6 1e-5
)
GRID_REDUCED_DIMS=(
    32 64 96
)
TARGET_BLOCKS="0|1|2|3|4|5|6|7|8|9|10|11"
TARGET_LAYERS="attn2.to_q|attn2.to_k|attn2.to_v"
TARGET_TAG="blk0-11_qkv"
FIXED_EPOCH=3
BOUNDS_DATASET_FRACTION=0.5

NUM_ALPHAS=${#GRID_ALPHAS[@]}
NUM_LAMBDAS=${#GRID_LAMBDAS[@]}
NUM_LRS=${#GRID_LRS[@]}
NUM_REDUCED_DIMS=${#GRID_REDUCED_DIMS[@]}
NUM_TUNED_COMBOS=$(( NUM_ALPHAS * NUM_LAMBDAS * NUM_LRS * NUM_REDUCED_DIMS ))

# Single-class search first: chain_saw (Imagenette class id 3).
CLASS_IDS=(3)
CLASS_NAMES=(
    "chain_saw"
)

NUM_CLASSES=${#CLASS_IDS[@]}          # 1
TOTAL_JOBS=$(( NUM_TUNED_COMBOS * NUM_CLASSES ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

COMBO_IDX=$(( TASK_ID / NUM_CLASSES ))
CLASS_SLOT=$(( TASK_ID % NUM_CLASSES ))
CLASS_ID=${CLASS_IDS[$CLASS_SLOT]}

# Decode COMBO_IDX into Cartesian product indices for (alpha, lambda, lr, reduced_dim)
RDM_IDX=$(( COMBO_IDX % NUM_REDUCED_DIMS ))
TMP_IDX=$(( COMBO_IDX / NUM_REDUCED_DIMS ))
LR_IDX=$(( TMP_IDX % NUM_LRS ))
TMP_IDX=$(( TMP_IDX / NUM_LRS ))
LAMBDA_IDX=$(( TMP_IDX % NUM_LAMBDAS ))
ALPHA_IDX=$(( TMP_IDX / NUM_LAMBDAS ))

ALPHA=${GRID_ALPHAS[$ALPHA_IDX]}
LAMBDA=${GRID_LAMBDAS[$LAMBDA_IDX]}
LR=${GRID_LRS[$LR_IDX]}
REDUCED_DIM=${GRID_REDUCED_DIMS[$RDM_IDX]}
EPOCH=${FIXED_EPOCH}

CLASS_NAME=${CLASS_NAMES[$CLASS_SLOT]}
PARAM_TAG="a${ALPHA}-lam${LAMBDA}-rdim${REDUCED_DIM}-ep${EPOCH}-lr${LR}"
SWEEP_KIND="chain_saw-allblocks-qkv-uafid-grid-v4"

echo "============================================"
echo "Grid search (${SWEEP_KIND}) – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS_ID} (${CLASS_NAME})"
echo "  Total active tasks: ${TOTAL_JOBS}"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
TMPCONFIG="/tmp/sd_grid_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"
RUN_ID="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
TMP_MODEL_DIR="/tmp/sd_grid_models/${RUN_ID}"
TMP_LOGS_DIR="/tmp/sd_grid_logs/${RUN_ID}"

python - "$CLASS_ID" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$TARGET_BLOCKS" "$TARGET_LAYERS" "$REDUCED_DIM" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" "$RUN_ID" "$TMP_MODEL_DIR" "$TMP_LOGS_DIR" "$TARGET_TAG" "$BOUNDS_DATASET_FRACTION" <<'PYEOF'
import yaml, sys

cls = int(sys.argv[1])
alpha_val = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5])
target_blocks = [int(s) for s in sys.argv[6].split("|") if s]
target_layers = [s for s in sys.argv[7].split("|") if s]
reduced_dim = int(sys.argv[8])
param_tag = sys.argv[9]
cls_name = sys.argv[10]
sweep_kind = sys.argv[11]
out = sys.argv[12]
run_id = sys.argv[13]
tmp_model_dir = sys.argv[14]
tmp_logs_dir = sys.argv[15]
target_tag = sys.argv[16]
bounds_dataset_fraction = float(sys.argv[17])

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["alpha"] = alpha_val
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["epochs"] = epochs
cfg["unlearn"]["save_compvis"] = True
cfg["unlearn"]["save_diffusers"] = True
cfg["unlearn"]["save_history_logs"] = False
cfg["intact"]["lambda_interval"] = lambda_val
cfg["intact"]["target_blocks"] = target_blocks
cfg["intact"]["target_layers"] = target_layers
cfg["intact"]["targets"] = [
    f"output_blocks.{block}.1.transformer_blocks.0.{layer}"
    for block in target_blocks
    for layer in target_layers
]
cfg["intact"]["reduced_dim"] = reduced_dim
cfg["intact"]["use_actual_bounds"] = True
cfg["intact"]["dataset_fraction"] = bounds_dataset_fraction

# Keep model artifacts off shared storage for this grid.
cfg["paths"]["model_save_dir"] = tmp_model_dir
cfg["paths"]["logs_dir"] = tmp_logs_dir

# Evaluation budget (same as class fulleval)
cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"] = 10
cfg["evaluate"]["fid"]["max_real"] = 900
cfg["evaluate"]["fid"]["max_fake"] = None

# wandb – grouped by combo for cross-class comparison
cfg["wandb"]["group"] = f"grid-{sweep_kind}-{param_tag}"
cfg["wandb"]["tags"] = [
    "sd", "class-wise", "intact", "fulleval", "grid-search", sweep_kind,
    f"alpha_{alpha_val}", f"lambda_{lambda_val}", f"epochs_{epochs}", f"lr_{lr}",
    f"targets_{target_tag}", f"rdim_{reduced_dim}", f"boundsfrac_{bounds_dataset_fraction}",
    cls_name,
]

# Unique output dir per (combo, class)
cfg["paths"]["output_dir"] = (
    cfg["paths"]["output_dir"] + f"/grid/{sweep_kind}/{param_tag}/class_{cls}/run_{run_id}"
)

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

# Cleanup temporary artifacts.
rm -rf "${TMP_MODEL_DIR}" "${TMP_LOGS_DIR}" "${TMPCONFIG}"

echo "${SWEEP_KIND}: combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS_ID} (${CLASS_NAME}) – done."

