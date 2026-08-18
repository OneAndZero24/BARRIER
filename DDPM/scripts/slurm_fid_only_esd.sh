#!/bin/bash
# ============================================================================
# SLURM – Train + FID-Only (5000/class) for ESD InTAct run (rerun)
# ============================================================================
# Trains the InTAct ESD (match-to-uncond) checkpoint from sweep:
#   lr=2.59e-4  n_iters=5000  bs=128  lambda=0.167  rd=32
#   use_actual_bounds=true  normalize_protection=true  esd_frozen_uncond=false
# then computes FID at reference-paper size (5000/class), no classifier.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_fid_only_esd.sh
# ============================================================================

#SBATCH --job-name=fid-only-esd
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# Redirect caches to avoid home quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

# ============================================================================
# Per-run hyperparameters (ESD sweep best config)
# ============================================================================
LABELS=(          "ESD lr0.000259 ni5000 lam0.167"   )
LRS=(             0.00025905313980398244              )
N_ITERS=(         5000                                 )
BATCH_SIZES=(     128                                  )
LAMBDAS=(         0.16700370314318672                  )
REDUCED_DIMS=(    32                                   )
USE_BOUNDS=(      true                                 )

# Shared
MODEL_CONFIG="configs/cifar10_intact.yml"
METHOD="esd"
MODE="intact"
LABEL_TO_FORGET=0
ALPHA=0.0
NEGATIVE_GUIDANCE=7.5
ESD_FROZEN_UNCOND=false
NORM_PROTECT=true
INF_SCALE=20
LOWER_PCT=0.05
UPPER_PCT=0.95
COND_SCALE=2.0
FID_N_SAMPLES=5000

echo "============================================"
echo "Train + FID-Only (ESD) – Job ${SLURM_JOB_ID}"
echo "  qos=big  mem=128GB  partition=dgxa100"
echo "  FID samples/class = ${FID_N_SAMPLES}"
echo "  Classifier = disabled"
echo "  1 run: ${LABELS[*]}"
echo "============================================"

for IDX in 0; do
    LABEL="${LABELS[$IDX]}"
    LR="${LRS[$IDX]}"
    NIT="${N_ITERS[$IDX]}"
    BS="${BATCH_SIZES[$IDX]}"
    LAM="${LAMBDAS[$IDX]}"
    RD="${REDUCED_DIMS[$IDX]}"
    UB="${USE_BOUNDS[$IDX]}"

    echo ""
    echo "============================================"
    echo ">>> Run $((IDX+1))/1 : ${LABEL}"
    echo "    method=${METHOD}  lr=${LR}  n_iters=${NIT}  bs=${BS}"
    echo "    lambda=${LAM}  reduced_dim=${RD}  use_bounds=${UB}"
    echo "============================================"

    TMPCONFIG="/tmp/fid_only_esd_${SLURM_JOB_ID}_${IDX}.yaml"

    python - <<PYEOF
import os, yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Unlearn ----
cfg["unlearn"]["mode"]              = "${MODE}"
cfg["unlearn"]["method"]            = "${METHOD}"
cfg["unlearn"]["label_to_forget"]   = ${LABEL_TO_FORGET}
cfg["unlearn"]["lr"]                = float("${LR}")
cfg["unlearn"]["n_iters"]           = int("${NIT}")
cfg["unlearn"]["batch_size"]        = int("${BS}")
cfg["unlearn"]["alpha"]             = ${ALPHA}
cfg["unlearn"]["negative_guidance"] = ${NEGATIVE_GUIDANCE}

# ---- Training (ESD-specific) ----
cfg.setdefault("training", {})["esd_frozen_uncond"] = "${ESD_FROZEN_UNCOND}".lower() == "true"

# ---- InTAct ----
cfg["intact"]["lambda_interval"]   = float("${LAM}")
cfg["intact"]["reduced_dim"]       = int("${RD}")
cfg["intact"]["use_actual_bounds"] = "${UB}".lower() == "true"
cfg["intact"]["normalize_protection"] = "${NORM_PROTECT}".lower() == "true"
cfg["intact"]["infinity_scale"]    = ${INF_SCALE}
cfg["intact"]["lower_percentile"]  = ${LOWER_PCT}
cfg["intact"]["upper_percentile"]  = ${UPPER_PCT}
cfg["intact"]["targets"] = [
    "attn.0.q",  "attn.0.k",  "attn.0.v",
    "attn_1.q",  "attn_1.k",  "attn_1.v",
    "attn.1.q",  "attn.1.k",  "attn.1.v",
    "cemb.dense.0", "cemb.dense.1",
]

# ---- Evaluate: FID 5000/class ONLY, no classifier ----
cfg["evaluate"]["fid"]["enabled"] = True
cfg["evaluate"]["fid"]["n_samples_per_class"] = ${FID_N_SAMPLES}
cfg["evaluate"]["classifier"]["enabled"] = False

# ---- Wandb ----
cfg["wandb"]["group"] = "cifar10-fid-only"
cfg["wandb"]["tags"] = ["fid-only", "5000", "${LABEL}"]

# Unique output suffix to avoid collisions
suffix = "esd_" + "${LABEL}".replace(" ", "_") + "_${SLURM_JOB_ID}_${IDX}"
cfg["paths"]["output_dir"]     = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

    echo "--- Running pipeline ---"
    python pipeline.py --config "${TMPCONFIG}"
    echo "--- Done: ${LABEL} ---"
done

echo ""
echo "============================================"
echo "All runs complete."
echo "============================================"
