#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search across ALL Imagenette classes
# NON-ZERO alpha only (new compositions)
# ============================================================================
#   Each array task runs one (hyperparam combo, class) pair.
#
#   Rationale for proposed compositions:
#   - Sweep a monotonic trend: larger alpha with smaller lambda.
#   - Keep LR modest as alpha grows to reduce instability.
#   - Increase epochs slightly for higher alpha settings.
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
#SBATCH --array=0-39

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- New NON-ZERO-alpha compositions ----
# Small grid: alpha values just above 0.5, with lambda on comparable scale.
# Keep lambda and alpha in the same order of magnitude to avoid over-regularizing.
TUNED_ALPHAS=(
    0.55 0.60 0.70 0.80
)
TUNED_LAMBDAS=(
    0.90 0.80 0.70 0.60
)
TUNED_EPOCHS=(
    4 4 5 5
)
TUNED_LRS=(
    6e-6 6e-6 5e-6 5e-6
)

NUM_TUNED_COMBOS=${#TUNED_ALPHAS[@]}  # 4
NUM_CLASSES=10
TOTAL_JOBS=$(( NUM_TUNED_COMBOS * NUM_CLASSES ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

CLASSES=("tench" "english_springer" "cassette_player" "chain_saw" "church"
         "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")

COMBO_IDX=$(( TASK_ID / NUM_CLASSES ))
CLASS=$(( TASK_ID % NUM_CLASSES ))

ALPHA=${TUNED_ALPHAS[$COMBO_IDX]}
LAMBDA=${TUNED_LAMBDAS[$COMBO_IDX]}
EPOCH=${TUNED_EPOCHS[$COMBO_IDX]}
LR=${TUNED_LRS[$COMBO_IDX]}

CLASS_NAME=${CLASSES[$CLASS]}
PARAM_TAG="a${ALPHA}-lam${LAMBDA}-ep${EPOCH}-lr${LR}"
SWEEP_KIND="nonzero-alpha-only-v3"

echo "============================================"
echo "Grid search (${SWEEP_KIND}) – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME})"
echo "  Total active tasks: ${TOTAL_JOBS}"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
TMPCONFIG="/tmp/sd_grid_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"
RUN_ID="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
TMP_MODEL_DIR="/tmp/sd_grid_models/${RUN_ID}"
TMP_LOGS_DIR="/tmp/sd_grid_logs/${RUN_ID}"

python - "$CLASS" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" "$RUN_ID" "$TMP_MODEL_DIR" "$TMP_LOGS_DIR" <<'PYEOF'
import yaml, sys

cls = int(sys.argv[1])
alpha_val = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5])
param_tag = sys.argv[6]
cls_name = sys.argv[7]
sweep_kind = sys.argv[8]
out = sys.argv[9]
run_id = sys.argv[10]
tmp_model_dir = sys.argv[11]
tmp_logs_dir = sys.argv[12]

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

echo "${SWEEP_KIND}: combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME}) – done."
