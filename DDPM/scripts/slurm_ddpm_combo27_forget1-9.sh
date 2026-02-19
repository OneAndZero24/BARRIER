#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM Combo 27 with Forget Classes 1-9
# ============================================================================
# Runs combo 27 (lr=1e-4, n_iters=3000, lambda=5.0, method=rl)
# with each class 1-9 as the forget class.
#
# HOW TO USE:
#   sbatch scripts/slurm_ddpm_combo27_forget1-9.sh
# ============================================================================

#SBATCH --job-name=ddpm-c27-f1-9
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-8

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Combo 27 parameters
# ============================================================================
COMBO_NUM=27
LR=1e-4
NITERS=3000
LAMBDA=5.0
METHOD="rl"

# Forget classes 1-9 (array index maps to forget class)
FORGET_CLASSES=(1 2 3 4 5 6 7 8 9)
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}
FORGET_CLASS=${FORGET_CLASSES[$IDX]}

echo "============================================"
echo "DDPM Combo 27 – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}:  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}  method=${METHOD}"
echo "  Forget Class: ${FORGET_CLASS}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/ddpm_combo27_fc${FORGET_CLASS}_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, copy, sys

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override hyperparams
cfg["unlearn"]["lr"]     = float("${LR}")
cfg["unlearn"]["n_iters"] = int("${NITERS}")
cfg["unlearn"]["method"] = "${METHOD}"
cfg["unlearn"]["label_to_forget"] = int("${FORGET_CLASS}")
cfg["intact"]["lambda_interval"] = float("${LAMBDA}")

# Tag the wandb run
cfg["wandb"]["tags"].append("combo27-forget1-9")
cfg["wandb"]["group"] = "cifar10-combo27"

# Unique output dirs per job
import os
suffix = f"combo{${COMBO_NUM}}_fc{${FORGET_CLASS}}_lr{${LR}}_ni{${NITERS}}_lam{${LAMBDA}}"
cfg["paths"]["output_dir"]     = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "DDPM Combo 27 – Forget Class ${FORGET_CLASS} – Job ${IDX} complete."
