#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search across ALL Imagenette classes
# NON-ZERO alpha only + expanded target blocks
# ============================================================================
#   Each array task runs one (hyperparam combo, class) pair.
#
#   Design goals:
#   - Avoid previously unstable high-LR / high-epoch regimes that spiked FID.
#   - Expand trainable/protected scope via explicit x-attn block targeting.
#   - Keep total jobs bounded so the full grid can finish within a week.
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
#SBATCH --array=0-35

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- Curated NON-ZERO-alpha compositions ----
# 6 combos x 10 classes = 60 active jobs.
# Use a compact selected-block target set, expanded in Python.
TUNED_ALPHAS=(
    0.12 0.18 0.22 0.25 0.18 0.30
)
TUNED_LAMBDAS=(
    4.0  4.5  3.0  2.5  5.5  2.0
)
TUNED_EPOCHS=(
    3 3 4 4 3 4
)
TUNED_LRS=(
    5e-6 4e-6 4e-6 3.5e-6 3.5e-6 3e-6
)
TARGET_BLOCKS="6|8"
TARGET_LAYERS="attn2.to_q|attn2.to_k|attn2.to_v|attn2.to_out.0"
TARGET_TAG="blk6-8_qkvo"
TUNED_REDUCED_DIM=(
    64 64 96 96 80 96
)
TUNED_LOWER_PCT=(
    0.05 0.05 0.08 0.10 0.08 0.10
)
TUNED_UPPER_PCT=(
    0.95 0.95 0.92 0.90 0.92 0.90
)
TUNED_INFINITY_SCALE=(
    20.0 18.0 15.0 12.0 14.0 12.0
)

NUM_TUNED_COMBOS=${#TUNED_ALPHAS[@]}  # 6

# Lagging classes only (drop classes with already strong UA/FID runs).
# Class IDs follow Imagenette mapping from pipeline config.
CLASS_IDS=(0 1 3 4 7 8)
CLASS_NAMES=(
    "tench"
    "english_springer"
    "chain_saw"
    "church"
    "gas_pump"
    "golf_ball"
)

NUM_CLASSES=${#CLASS_IDS[@]}          # 6
TOTAL_JOBS=$(( NUM_TUNED_COMBOS * NUM_CLASSES ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

COMBO_IDX=$(( TASK_ID / NUM_CLASSES ))
CLASS_SLOT=$(( TASK_ID % NUM_CLASSES ))
CLASS_ID=${CLASS_IDS[$CLASS_SLOT]}

ALPHA=${TUNED_ALPHAS[$COMBO_IDX]}
LAMBDA=${TUNED_LAMBDAS[$COMBO_IDX]}
EPOCH=${TUNED_EPOCHS[$COMBO_IDX]}
LR=${TUNED_LRS[$COMBO_IDX]}
REDUCED_DIM=${TUNED_REDUCED_DIM[$COMBO_IDX]}
LOWER_PCT=${TUNED_LOWER_PCT[$COMBO_IDX]}
UPPER_PCT=${TUNED_UPPER_PCT[$COMBO_IDX]}
INF_SCALE=${TUNED_INFINITY_SCALE[$COMBO_IDX]}

CLASS_NAME=${CLASS_NAMES[$CLASS_SLOT]}
PARAM_TAG="a${ALPHA}-lam${LAMBDA}-ep${EPOCH}-lr${LR}"
SWEEP_KIND="nonzero-alpha-blockgrid-v1"

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

python - "$CLASS_ID" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$TARGET_BLOCKS" "$TARGET_LAYERS" "$REDUCED_DIM" "$LOWER_PCT" "$UPPER_PCT" "$INF_SCALE" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" "$RUN_ID" "$TMP_MODEL_DIR" "$TMP_LOGS_DIR" "$TARGET_TAG" <<'PYEOF'
import yaml, sys

cls = int(sys.argv[1])
alpha_val = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5])
target_blocks = [int(s) for s in sys.argv[6].split("|") if s]
target_layers = [s for s in sys.argv[7].split("|") if s]
reduced_dim = int(sys.argv[8])
lower_pct = float(sys.argv[9])
upper_pct = float(sys.argv[10])
infinity_scale = float(sys.argv[11])
param_tag = sys.argv[12]
cls_name = sys.argv[13]
sweep_kind = sys.argv[14]
out = sys.argv[15]
run_id = sys.argv[16]
tmp_model_dir = sys.argv[17]
tmp_logs_dir = sys.argv[18]
target_tag = sys.argv[19]

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
cfg["intact"]["lower_percentile"] = lower_pct
cfg["intact"]["upper_percentile"] = upper_pct
cfg["intact"]["infinity_scale"] = infinity_scale

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
    f"targets_{target_tag}", f"rdim_{reduced_dim}",
    f"pct_{lower_pct}_{upper_pct}", f"inf_{infinity_scale}",
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

