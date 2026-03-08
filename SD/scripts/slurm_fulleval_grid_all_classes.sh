#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search across ALL Imagenette classes
# ============================================================================
#   Fully parallelised: each job = one (hyperparam combo, class) pair.
#   Total jobs = NUM_COMBOS × 10 classes.
#   wandb groups by combo so you can average UA/FID across classes.
#
#   Indexing:  COMBO_IDX = TASK_ID / 10,  CLASS = TASK_ID % 10
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
#SBATCH --array=0-79

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- Hyperparameter grid (8 combos) ----
LAMBDAS=(5    10   1    0.5  10   5    5    1   )
EPOCHS=( 3    3    2    2    5    2    5    5   )
LRS=(    5e-6 5e-6 5e-6 5e-6 5e-6 1e-5 1e-5 1e-5)

NUM_COMBOS=${#LAMBDAS[@]}   # 8
NUM_CLASSES=10

# ---- 2-D indexing ----
TASK_ID=${SLURM_ARRAY_TASK_ID}
COMBO_IDX=$(( TASK_ID / NUM_CLASSES ))
CLASS=$(( TASK_ID % NUM_CLASSES ))

LAMBDA=${LAMBDAS[$COMBO_IDX]}
EPOCH=${EPOCHS[$COMBO_IDX]}
LR=${LRS[$COMBO_IDX]}

CLASSES=("tench" "english_springer" "cassette_player" "chain_saw" "church"
         "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")
CLASS_NAME=${CLASSES[$CLASS]}

PARAM_TAG="lam${LAMBDA}-ep${EPOCH}-lr${LR}"

echo "============================================"
echo "Grid search – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME})"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
TMPCONFIG="/tmp/sd_grid_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"

python - "$CLASS" "$LAMBDA" "$EPOCH" "$LR" "$PARAM_TAG" "$CLASS_NAME" "$TMPCONFIG" <<'PYEOF'
import yaml, sys

cls        = int(sys.argv[1])
lambda_val = float(sys.argv[2])
epochs     = int(sys.argv[3])
lr         = float(sys.argv[4])
param_tag  = sys.argv[5]
cls_name   = sys.argv[6]
out        = sys.argv[7]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["lr"]              = lr
cfg["unlearn"]["epochs"]          = epochs
cfg["intact"]["lambda_interval"]  = lambda_val

# Evaluation budget (same as single-class fulleval)
cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"]                = 10
cfg["evaluate"]["fid"]["max_real"]        = 900
cfg["evaluate"]["fid"]["max_fake"]        = None

# wandb – group by hyperparam combo so cross-class averages are trivial
cfg["wandb"]["group"] = f"grid-{param_tag}"
cfg["wandb"]["tags"]  = [
    "sd", "class-wise", "intact", "fulleval", "grid-search",
    f"lambda_{lambda_val}", f"epochs_{epochs}", f"lr_{lr}",
    cls_name,
]

# Unique output dir per (combo, class)
cfg["paths"]["output_dir"] = (
    cfg["paths"]["output_dir"] + f"/grid/{param_tag}/class_{cls}"
)

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "Combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME}) – done."
