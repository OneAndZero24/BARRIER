#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 InTAct GA Grid (Gradient Ascent)
# ============================================================================
# Grid over lambda_interval values.  Fixed: lr=1e-4, n_iters=3000, reduced_dim=32,
# targets: Self-attn QKV + class_embed.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_ga_grid.sh
# ============================================================================

#SBATCH --job-name=ddpm-ga-grid
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
METHOD="ga"
USE_ACTUAL_BOUNDS=true
REDUCED_DIM=32
NORMALIZE_PROTECTION=true
INF_SCALE=20.0
LOWER=0.05
UPPER=0.95

# ---- Grid: lambda_interval sweep ----
LAMBDAS=(0.5 1.0 5.0 10.0 50.0)

IDX=${SLURM_ARRAY_TASK_ID}
LAMBDA=${LAMBDAS[$IDX]}

echo "============================================"
echo "DDPM GA Grid – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  forget_class=${FORGET_CLASS}  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "  method=${METHOD}  reduced_dim=${REDUCED_DIM}"
echo "  targets: Self-attn QKV + class_embed"
echo "============================================"

TMPCONFIG="/tmp/ddpm_ga_grid_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

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
cfg["intact"]["targets"] = [
    "attn.0.q", "attn.0.k", "attn.0.v",
    "attn_1.q", "attn_1.k", "attn_1.v",
    "attn.1.q", "attn.1.k", "attn.1.v",
    "cemb.dense.0", "cemb.dense.1",
]

cfg.setdefault("evaluate", {}).setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

cfg.setdefault("wandb", {})
cfg["wandb"]["group"] = "cifar10-ga-qkv-cemb-grid"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "ga", "ablation", "qkv-classemb-grid",
    f"lambda_{lam}", "lr_1e-4", "iters_3000",
]

suffix = f"ga_qkv_cemb_lam{lam}_lr{lr}_ni{niters}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "GA grid job ${IDX} (lambda=${LAMBDA}) complete."
