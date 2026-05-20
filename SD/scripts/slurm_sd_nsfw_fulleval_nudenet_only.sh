#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW Full Evaluation (NudeNet only)
# ============================================================================
# Runs the SD NSFW unlearning pipeline for the selected hyperparameter combo(s):
#   Unlearn → Generate 4703 I2P images (1 per prompt) → NudeNet I2P (thr=0.6)
#
# COCO FID / CLIP / generation are disabled in this variant.
#
# HOW TO USE:
#   1. Fill in the HPARAM arrays below with the combo you want to rerun
#   2. Set --array=0-<N-1> where N = number of hparam combos
#   3. sbatch scripts/slurm_sd_nsfw_fulleval_nudenet_only.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-nudenet
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-7

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Hyperparameter combinations to rerun (8-combo grid)
# ============================================================================
# Exploring: LR in {1e-6, 5e-6, 1e-5} x EPOCHS in {3, 5} x LAMBDA in {0.5, 1.0}
COMBO_NUMBERS=(     1         2         3         4         5         6         7         8       )
LEARNING_RATES=(    1e-6      1e-6      5e-6      5e-6      1e-5      1e-5      5e-6      1e-5     )
EPOCHS=(            3         5         3         5         3         5         5         3        )
LAMBDA_INTERVALS=(  0.5       1.0       0.5       1.0       0.5       1.0       1.0       1.0      )
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}

COMBO_NUM=${COMBO_NUMBERS[$IDX]}
LR=${LEARNING_RATES[$IDX]}
EP=${EPOCHS[$IDX]}
LAMBDA=${LAMBDA_INTERVALS[$IDX]}

echo "============================================"
echo "SD NSFW NudeNet-only Rerun – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}:  lr=${LR}  epochs=${EP}  lambda=${LAMBDA}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/sd_nsfw_nudenet_only_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override hyperparams
cfg["unlearn"]["lr"] = float("${LR}")
cfg["unlearn"]["epochs"] = int("${EP}")
cfg["intact"]["lambda_interval"] = float("${LAMBDA}")

# Disable COCO generation / FID / CLIP and probe logging for this rerun
cfg["evaluate"]["coco"]["enabled"] = False
cfg["evaluate"]["fid"]["enabled"] = False
cfg["evaluate"]["probe"]["enabled"] = False
cfg["evaluate"]["nudenet"]["enabled"] = True

# Tag the wandb run
cfg["wandb"]["tags"].append("nudenet-only")
cfg["wandb"]["group"] = "nsfw-fulleval-nudenet-only"

# Unique output dir per job
suffix = f"combo{${COMBO_NUM}}_lr{${LR}}_ep{${EP}}_lam{${LAMBDA}}_nudenet_only"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "SD NSFW NudeNet-only Rerun – Job ${IDX} complete."