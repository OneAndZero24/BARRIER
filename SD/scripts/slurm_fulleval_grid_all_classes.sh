#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search across ALL Imagenette classes
# ============================================================================
#   Fully parallelised: each job = one (hyperparam combo, class) pair.
#
#   This script runs two NON-ZERO-alpha stages in one array:
#   1) RETRY of previously failed/crashed jobs.
#   2) REFINED sweep across all 10 Imagenette classes with tighter ranges
#      around the currently best UA/FID trade-off region.
#
#   Indexing:
#     - TASK_ID < RETRY_JOBS         -> retry non-zero-alpha failures
#     - otherwise                    -> refined non-zero-alpha sweep
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_grid_all_classes.sh
# ============================================================================

#SBATCH --job-name=sd-grid-all
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-63

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- Retry set: failed/crashed NON-ZERO-alpha jobs from prior run ----
RETRY_CLASSES=( 0    2    5    9 )
RETRY_ALPHAS=( 0.10 0.05 0.05 0.05 )
RETRY_LAMBDAS=(5    10   10   10 )
RETRY_EPOCHS=( 3    2    2    2 )
RETRY_LRS=(    5e-6 1e-5 1e-5 1e-5 )

NUM_RETRY_JOBS=${#RETRY_CLASSES[@]}  # 4

# ---- Refined NON-ZERO-alpha grid across all classes ----
# Keep grid compact, but cover low + medium + high alpha to avoid bias.
TUNED_ALPHAS=( 0.05 0.05 0.10 0.10 0.20 0.30 )
TUNED_LAMBDAS=(5    10   5    10   7    3   )
TUNED_EPOCHS=( 2    2    3    3    4    5   )
TUNED_LRS=(    5e-6 1e-5 5e-6 1e-5 1e-5 5e-6)

NUM_TUNED_COMBOS=${#TUNED_ALPHAS[@]}  # 6
NUM_CLASSES=10

RETRY_JOBS=${NUM_RETRY_JOBS}
TUNED_JOBS=$(( NUM_TUNED_COMBOS * NUM_CLASSES ))
TOTAL_JOBS=$(( RETRY_JOBS + TUNED_JOBS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

CLASSES=("tench" "english_springer" "cassette_player" "chain_saw" "church"
         "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")

if (( TASK_ID < RETRY_JOBS )); then
    SWEEP_KIND="retry-nonzero-failures"
    LOCAL_ID=${TASK_ID}
    COMBO_IDX=${LOCAL_ID}
    CLASS=${RETRY_CLASSES[$LOCAL_ID]}

    ALPHA=${RETRY_ALPHAS[$LOCAL_ID]}
    LAMBDA=${RETRY_LAMBDAS[$LOCAL_ID]}
    EPOCH=${RETRY_EPOCHS[$LOCAL_ID]}
    LR=${RETRY_LRS[$LOCAL_ID]}
else
    SWEEP_KIND="tuned-nonzero-alpha"
    LOCAL_ID=$(( TASK_ID - RETRY_JOBS ))
    COMBO_IDX=$(( LOCAL_ID / NUM_CLASSES ))
    CLASS=$(( LOCAL_ID % NUM_CLASSES ))

    ALPHA=${TUNED_ALPHAS[$COMBO_IDX]}
    LAMBDA=${TUNED_LAMBDAS[$COMBO_IDX]}
    EPOCH=${TUNED_EPOCHS[$COMBO_IDX]}
    LR=${TUNED_LRS[$COMBO_IDX]}
fi

CLASS_NAME=${CLASSES[$CLASS]}
PARAM_TAG="a${ALPHA}-lam${LAMBDA}-ep${EPOCH}-lr${LR}"

echo "============================================"
echo "Grid search (${SWEEP_KIND}) – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME})"
echo "  Total active tasks: ${TOTAL_JOBS} (retry=${RETRY_JOBS}, tuned=${TUNED_JOBS})"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
TMPCONFIG="/tmp/sd_grid_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"
RUN_ID="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
TMP_MODEL_DIR="/tmp/sd_grid_models/${RUN_ID}"
TMP_LOGS_DIR="/tmp/sd_grid_logs/${RUN_ID}"

python - "$CLASS" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" "$RUN_ID" "$TMP_MODEL_DIR" "$TMP_LOGS_DIR" <<'PYEOF'
import yaml, sys

cls        = int(sys.argv[1])
alpha_val  = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs     = int(sys.argv[4])
lr         = float(sys.argv[5])
param_tag  = sys.argv[6]
cls_name   = sys.argv[7]
sweep_kind = sys.argv[8]
out        = sys.argv[9]
run_id     = sys.argv[10]
tmp_model_dir = sys.argv[11]
tmp_logs_dir = sys.argv[12]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["alpha"]           = alpha_val
cfg["unlearn"]["lr"]              = lr
cfg["unlearn"]["epochs"]          = epochs
# Diffusers export still reloads the saved compvis checkpoint, so keep it on
# until conversion finishes. The temp directory is removed at the end anyway.
cfg["unlearn"]["save_compvis"]    = True
cfg["unlearn"]["save_diffusers"]  = True
cfg["unlearn"]["save_history_logs"] = False
cfg["intact"]["lambda_interval"]  = lambda_val

# Keep model artifacts off shared storage for this grid.
cfg["paths"]["model_save_dir"] = tmp_model_dir
cfg["paths"]["logs_dir"] = tmp_logs_dir

# Evaluation budget (same as single-class fulleval)
cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"]                = 10
cfg["evaluate"]["fid"]["max_real"]        = 900
cfg["evaluate"]["fid"]["max_fake"]        = None

# wandb – group by hyperparam combo so cross-class averages are trivial
cfg["wandb"]["group"] = f"grid-{sweep_kind}-{param_tag}"
cfg["wandb"]["tags"]  = [
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

# Cleanup temporary model/log artifacts to avoid filling local disks over time.
rm -rf "${TMP_MODEL_DIR}" "${TMP_LOGS_DIR}" "${TMPCONFIG}"

echo "${SWEEP_KIND}: combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME}) – done."
