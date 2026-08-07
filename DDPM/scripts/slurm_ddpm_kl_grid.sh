#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 InTAct KL Div Grid (KL Divergence)
# ============================================================================
# Full hyperparameter sweep:
#   lr              in {5e-5, 1e-4, 5e-4, 1e-3}
#   n_iters         in {1000, 3000, 5000}
#   lambda_interval in {0.5, 1.0, 5.0, 10.0, 50.0}
#   reduced_dim     in {16, 32, 64, 128}
#
# Total: 4 × 1 × 5 × 2 = 40 jobs
# Fixed:  method=kl, targets=QKV+cemb, use_actual_bounds=true,
#         lower=0.05, upper=0.95, inf_scale=20.0, norm_prot=true
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_kl_grid.sh
# ============================================================================

#SBATCH --job-name=ddpm-kl-grid
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-39

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

# ---- Fixed setup ----
FORGET_CLASS=0
METHOD="kl"
USE_ACTUAL_BOUNDS=true
NORMALIZE_PROTECTION=true
INF_SCALE=20.0
LOWER=0.05
UPPER=0.95

# ---- Grid axes ----
LRS=(5e-5 1e-4 5e-4 1e-3)       # 4
NITERS_VALS=(3000)                # 1
LAMBDAS=(0.1 0.5 1.0 5.0 10.0)  # 5
REDUCED_DIMS=(8 32)               # 2

# ---- Index mapping: IDX = lidx*(1*5*2) + nidx*(5*2) + lamidx*2 + didx ----
IDX=${SLURM_ARRAY_TASK_ID}

N_NITERS=1
N_LAMBDA=5
N_RDIM=2
STRIDE_NITERS=$(( N_NITERS * N_LAMBDA * N_RDIM ))   # 10
STRIDE_LAMBDA=$(( N_LAMBDA * N_RDIM ))               # 10
STRIDE_RDIM=$(( N_RDIM ))                            # 2

LIDX=$(( IDX / STRIDE_NITERS ))
R1=$(( IDX % STRIDE_NITERS ))
NIDX=$(( R1 / STRIDE_LAMBDA ))
R2=$(( R1 % STRIDE_LAMBDA ))
LAMIDX=$(( R2 / STRIDE_RDIM ))
DIDX=$(( R2 % STRIDE_RDIM ))

LR=${LRS[$LIDX]}
NITERS=${NITERS_VALS[$NIDX]}
LAMBDA=${LAMBDAS[$LAMIDX]}
REDUCED_DIM=${REDUCED_DIMS[$DIDX]}

echo "============================================"
echo "DDPM KL Grid – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}  reduced_dim=${REDUCED_DIM}"
echo "  method=${METHOD}  forget_class=${FORGET_CLASS}"
echo "============================================"

TMPCONFIG="/tmp/ddpm_kl_grid_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

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
cfg["wandb"]["group"] = "cifar10-kl-qkv-cemb-fullsweep"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "kl", "fullsweep", "qkv-classemb",
    f"lr_{lr}", f"niters_{niters}", f"lambda_{lam}", f"rdim_{reduced_dim}",
]

suffix = f"kl_lr{lr}_ni{niters}_lam{lam}_rdim{reduced_dim}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "KL grid job ${IDX} (lr=${LR}, n_iters=${NITERS}, lambda=${LAMBDA}, rdim=${REDUCED_DIM}) complete."
