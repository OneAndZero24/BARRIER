#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 Class-wise Forgetting (Table 3)
# ============================================================================
# Goal:
#   Run class-wise forgetting on CIFAR-10 for the five classes reported in the
#   paper table: automobile, cat, dog, horse, truck.
#
# Current best baseline to seed the grid:
#   - Self-attn QKV + class embedding targets
#   - k=32 (reduced_dim)
#   - lambda=5
#   - Adam optimizer
#   - lr=1e-3
#   - 3000 steps
#
# The pipeline writes metrics.json into each run directory. After the jobs
# finish, use scripts/collect_ddpm_classwise_table3.py to print the table.
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_classwise_table3.sh
# ============================================================================

#SBATCH --job-name=ddpm-table3
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

# ---- Table 3 classes ----
FORGET_CLASSES=(1 3 5 7 9)

# ---- Best baseline hyperparameters ----
MODEL_CONFIG="configs/cifar10_intact_qkv.yml"
LR=1e-3
NITERS=3000
LAMBDA=5.0
REMAIN_BOUNDS=true
REDUCED_DIM=32
LOWER=0.05
UPPER=0.95
METHOD="rl"

IDX=${SLURM_ARRAY_TASK_ID}
FORGET_CLASS=${FORGET_CLASSES[$IDX]}
REF_DATASET_DIR="./cifar10_without_label_${FORGET_CLASS}"

if [[ ! -d "${REF_DATASET_DIR}" ]]; then
    echo "Creating reference dataset at ${REF_DATASET_DIR}"
    python save_base_dataset.py --dataset cifar10 --label_to_forget "${FORGET_CLASS}"
fi

echo "============================================"
echo "DDPM Table 3 Class-wise Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  forget_class=${FORGET_CLASS}"
echo "  model_config=${MODEL_CONFIG}"
echo "  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "============================================"

TMPCONFIG="/tmp/ddpm_table3_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import os
import yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

forget_class = int("${FORGET_CLASS}")
cfg["model_config"] = "${MODEL_CONFIG}"
cfg["unlearn"]["label_to_forget"] = forget_class
cfg["unlearn"]["lr"] = float("${LR}")
cfg["unlearn"]["n_iters"] = int("${NITERS}")
cfg["unlearn"]["method"] = "${METHOD}"

cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = float("${LAMBDA}")
cfg["intact"]["use_actual_bounds"] = "${REMAIN_BOUNDS}".lower() == "true"
cfg["intact"]["reduced_dim"] = int("${REDUCED_DIM}")
cfg["intact"]["lower_percentile"] = float("${LOWER}")
cfg["intact"]["upper_percentile"] = float("${UPPER}")
cfg["intact"]["targets"] = [
    "attn.0.q",
    "attn.0.k",
    "attn.0.v",
    "attn_1.q",
    "attn_1.k",
    "attn_1.v",
    "attn.1.q",
    "attn.1.k",
    "attn.1.v",
    "cemb.dense.0",
    "cemb.dense.1",
]

cfg["paths"]["ref_dataset_dir"] = os.path.join(".", f"cifar10_without_label_{forget_class}")
suffix = f"table3_fc{forget_class}_lr${{LR}}_ni${{NITERS}}_lam${{LAMBDA}}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

cfg.setdefault("wandb", {})
cfg["wandb"]["group"] = "cifar10-table3-classwise"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "table3",
    f"forget_{forget_class}",
    "qkv+cemb",
    "k32",
    "lambda5",
    "adam",
    "lr1e-3",
    "iters3000",
]

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "DDPM Table 3 class-wise job ${IDX} complete."