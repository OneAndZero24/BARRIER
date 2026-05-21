#!/bin/bash
# ============================================================================
# SLURM Array Job – SD class forgetting big grid across ALL Imagenette classes
# ============================================================================
#   Alpha is fixed at 0.0 and we sweep several hyperparameter settings across
#   all 10 Imagenette classes.
#
#   This is helios-ready and follows the Flux-style SLURM/runtime convention:
#   - plgrid-gpu-gh200 partition
#   - ML-bundle/25.10 module
#   - SCRATCH-aware cache root with home fallback
#
#   Resume behavior:
#   - Re-run the same array job after timeout/crash.
#   - Output paths are stable per (class, hyperparameter combo), so completed
#     images remain on disk and the generator skips them on the next run.
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_grid_all_classes.sh
# ============================================================================

#SBATCH --job-name=sd-grid-all-a0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-159

set -euo pipefail

# ---- Environment ----
ml ML-bundle/25.10
VENV_ROOT="${VENV_ROOT:-$SCRATCH/sd_venv}"
if [ ! -f "$VENV_ROOT/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $VENV_ROOT"
    echo "Create it with: ml ML-bundle/25.10 && python3.9 -m venv \"$VENV_ROOT\""
    exit 1
fi
source "$VENV_ROOT/bin/activate"
cd "$HOME/InTAct-Unl/SD"
export PYTHONPATH="$HOME/InTAct-Unl:${PYTHONPATH:-}"

if [ -n "${SCRATCH:-}" ]; then
    CACHE_BASE="$SCRATCH/.cache"
    RESULTS_ROOT="$SCRATCH/intact/SD"
else
    CACHE_BASE="$HOME/.cache/intact"
    RESULTS_ROOT="$HOME/results/intact/SD"
fi

export CACHE_ROOT="$CACHE_BASE"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export CLIP_CACHE_DIR="$CACHE_ROOT/clip"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$CACHE_ROOT" "$WANDB_DIR" "$TMPDIR" "$RESULTS_ROOT"

# ---- Alpha-zero grid across all classes ----
# 4 lambdas x 2 epochs x 2 learning rates x 2 reduced_dims = 16 combos.
# With 10 Imagenette classes, the total grid size is 160 jobs.
GRID_LAMBDAS=(0.5 1 5 10)
GRID_EPOCHS=(2 3)
GRID_LRS=(5e-6 1e-5)
GRID_REDUCED_DIMS=(64 96)
TARGET_BLOCKS=(0 1 2 3 4 5 6 7 8 9 10 11)
TARGET_LAYERS=("attn2.to_q" "attn2.to_k" "attn2.to_v")
CLASS_NAMES=("tench" "english_springer" "cassette_player" "chain_saw" "church" \
             "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")

NUM_LAMBDAS=${#GRID_LAMBDAS[@]}
NUM_EPOCHS=${#GRID_EPOCHS[@]}
NUM_LRS=${#GRID_LRS[@]}
NUM_REDUCED_DIMS=${#GRID_REDUCED_DIMS[@]}
NUM_CLASSES=${#CLASS_NAMES[@]}
NUM_COMBOS=$(( NUM_LAMBDAS * NUM_EPOCHS * NUM_LRS * NUM_REDUCED_DIMS ))
TOTAL_JOBS=$(( NUM_COMBOS * NUM_CLASSES ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

COMBO_IDX=$(( TASK_ID / NUM_CLASSES ))
CLASS=$(( TASK_ID % NUM_CLASSES ))

TMP_IDX=${COMBO_IDX}
RDM_IDX=$(( TMP_IDX % NUM_REDUCED_DIMS ))
TMP_IDX=$(( TMP_IDX / NUM_REDUCED_DIMS ))
LR_IDX=$(( TMP_IDX % NUM_LRS ))
TMP_IDX=$(( TMP_IDX / NUM_LRS ))
EPOCH_IDX=$(( TMP_IDX % NUM_EPOCHS ))
LAMBDA_IDX=$(( TMP_IDX / NUM_EPOCHS ))

ALPHA=0.0
LAMBDA=${GRID_LAMBDAS[$LAMBDA_IDX]}
EPOCH=${GRID_EPOCHS[$EPOCH_IDX]}
LR=${GRID_LRS[$LR_IDX]}
REDUCED_DIM=${GRID_REDUCED_DIMS[$RDM_IDX]}

CLASS_NAME=${CLASS_NAMES[$CLASS]}
PARAM_TAG="a0-lam${LAMBDA}-ep${EPOCH}-lr${LR}-rdim${REDUCED_DIM}"
SWEEP_KIND="alpha0-allclasses-biggrid-helios-v1"

echo "============================================"
echo "Grid search (${SWEEP_KIND}) – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME})"
echo "  Total active tasks: ${TOTAL_JOBS} (combos=${NUM_COMBOS}, classes=${NUM_CLASSES})"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
RUN_ID="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
TMP_ROOT="${TMPDIR}/sd_grid/${RUN_ID}"
TMPCONFIG="${TMP_ROOT}/config.yaml"
MODEL_DIR="${RESULTS_ROOT}/models/${SWEEP_KIND}/${PARAM_TAG}/class_${CLASS}"
LOGS_DIR="${RESULTS_ROOT}/logs/${SWEEP_KIND}/${PARAM_TAG}/class_${CLASS}"
mkdir -p "$TMP_ROOT" "$MODEL_DIR" "$LOGS_DIR"

python - "$CLASS" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$REDUCED_DIM" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" "$RUN_ID" "$MODEL_DIR" "$LOGS_DIR" "$RESULTS_ROOT" <<'PYEOF'
import yaml, sys
from pathlib import Path

cls = int(sys.argv[1])
alpha_val = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5])
reduced_dim = int(sys.argv[6])
param_tag = sys.argv[7]
cls_name = sys.argv[8]
sweep_kind = sys.argv[9]
out = sys.argv[10]
run_id = sys.argv[11]
model_dir = sys.argv[12]
logs_dir = sys.argv[13]
results_root = sys.argv[14]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["alpha"] = alpha_val
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["epochs"] = epochs
cfg["unlearn"]["save_compvis"] = True
cfg["unlearn"]["save_diffusers"] = True
cfg["unlearn"]["save_history_logs"] = False

cfg["intact"]["lambda_interval"] = lambda_val
cfg["intact"]["target_blocks"] = list(range(12))
cfg["intact"]["target_layers"] = ["attn2.to_q", "attn2.to_k", "attn2.to_v"]
cfg["intact"]["targets"] = [
    f"output_blocks.{block}.1.transformer_blocks.0.{layer}"
    for block in range(12)
    for layer in ["attn2.to_q", "attn2.to_k", "attn2.to_v"]
]
cfg["intact"]["reduced_dim"] = reduced_dim
cfg["intact"]["use_actual_bounds"] = True
cfg["intact"]["dataset_fraction"] = 1.0

cfg["paths"]["model_save_dir"] = model_dir
cfg["paths"]["logs_dir"] = logs_dir
cfg["paths"]["output_dir"] = f"{results_root}/grid/{sweep_kind}/{param_tag}/class_{cls}"

model_name = (
    f"compvis-intact-rl-class_{cls}-targets_blk0-11_qkv"
    f"-lambda_{lambda_val}-epochs_{epochs}-lr_{lr}"
)
checkpoint_path = Path(model_dir) / model_name / f"{model_name.replace('compvis', 'diffusers')}.pt"
if checkpoint_path.exists():
    cfg["pipeline"]["eval_only"] = True
    cfg["pipeline"]["model_name"] = model_name

cfg["evaluate"]["num_samples_per_prompt"] = 50
cfg["evaluate"]["n_outer"] = 10
cfg["evaluate"]["fid"]["enabled"] = False

cfg["wandb"]["group"] = f"grid-{sweep_kind}-{param_tag}"
cfg["wandb"]["tags"] = [
    "sd", "class-wise", "intact", "fulleval", "grid-search", sweep_kind,
    f"alpha_{alpha_val}", f"lambda_{lambda_val}", f"epochs_{epochs}", f"lr_{lr}",
    f"rdim_{reduced_dim}", cls_name,
]

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "$TMPCONFIG"

# Cleanup temporary job artifacts.
rm -rf "$TMP_ROOT"

echo "${SWEEP_KIND}: combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME}) – done."
