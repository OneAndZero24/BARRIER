#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 InTAct Ablation: no-actual-bounds grid
# ============================================================================
# Goal:
#   Disable remain-set-based ACTUAL bounds and search interval hyperparameters.
#
# Fixed hyperparameters:
#   lr=1e-4, n_iters=3000, lambda_interval=5.0, method=rl, use_actual_bounds=false
#
# Grid axis:
#   infinity_scale (many values)
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_ablation_no_actual_bounds_grid.sh
# ============================================================================

#SBATCH --job-name=ddpm-nobounds-grid
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-4

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ---- Fixed setup ----
FORGET_CLASS=0
LR=1e-4
NITERS=3000
LAMBDA=5.0
METHOD="rl"
USE_ACTUAL_BOUNDS=false
REDUCED_DIM=32
NORMALIZE_PROTECTION=true

# ---- Grid: infinity_scale sweep (10 values) ----
LOWER=0.05
UPPER=0.95
INFTY_SCALES=(1 10 1000 1000000 1000000000)

IDX=${SLURM_ARRAY_TASK_ID}
INF_SCALE=${INFTY_SCALES[$IDX]}

echo "============================================"
echo "DDPM No-Actual-Bounds Grid – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  forget_class=${FORGET_CLASS}  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "  method=${METHOD}  use_actual_bounds=${USE_ACTUAL_BOUNDS}"
echo "  lower=${LOWER}  upper=${UPPER}  infinity_scale=${INF_SCALE} (grid index ${IDX})"
echo "============================================"

TMPCONFIG="/tmp/ddpm_ablate_nobounds_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import os
import yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

forget_class = int("${FORGET_CLASS}")
lr = float("${LR}")
niters = int("${NITERS}")
lam = float("${LAMBDA}")
method = "${METHOD}"
use_actual_bounds = "${USE_ACTUAL_BOUNDS}".lower() == "true"
lower = float("${LOWER}")
upper = float("${UPPER}")
inf_scale = float("${INF_SCALE}")
reduced_dim = int("${REDUCED_DIM}")
normalize_protection = "${NORMALIZE_PROTECTION}".lower() == "true"

cfg["unlearn"]["label_to_forget"] = forget_class
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["n_iters"] = niters
cfg["unlearn"]["method"] = method

cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = lam
cfg["intact"]["use_actual_bounds"] = use_actual_bounds
cfg["intact"]["lower_percentile"] = lower
cfg["intact"]["upper_percentile"] = upper
cfg["intact"]["infinity_scale"] = inf_scale
cfg["intact"]["reduced_dim"] = reduced_dim
cfg["intact"]["normalize_protection"] = normalize_protection

# Keep full-eval budget (5k FID / 500 classifier)
cfg.setdefault("evaluate", {}).setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

cfg.setdefault("wandb", {})
cfg["wandb"]["group"] = "cifar10-ablate-nobounds-infscale-grid"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "ablation", "no-actual-bounds", "infscale-grid",
    f"infscale_{inf_scale}", f"lower_0.05", f"upper_0.95",
    "lr_1e-4", "iters_3000", "lambda_5.0"
]

suffix = (
    f"ablate_nobounds_infscale_{inf_scale}"
    f"_lr{lr}_ni{niters}_lam{lam}"
)
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "No-actual-bounds grid job ${IDX} complete."
