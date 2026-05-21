#!/bin/bash
# ============================================================================
# SLURM – SD NSFW Eval-Only COCO30k + 5-sample probe (pre-trained checkpoint)
# ============================================================================
# Evaluates a trained NSFW checkpoint without retraining:
#   - reuses the existing I2P/generated images for NudeNet evaluation
#   - runs full COCO30k generation + FID + CLIP
#   - generates 5 nude-vs-clothed probe samples for the trained checkpoint
# ============================================================================

#SBATCH --job-name=sd-nsfw-coco30k-p5
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxh100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---- Trained checkpoint to evaluate ----
MODEL_NAME="compvis-intact-nsfw-targets_tgth_675706798c_n3-lambda_0.5-lr_5e-06"
MODEL_PATH="/shared/results/common/miksa/intact/SD/models/${MODEL_NAME}/diffusers-intact-nsfw-targets_tgth_675706798c_n3-lambda_0.5-lr_5e-06.pt"

# ---- Existing I2P images from the trained run ----
I2P_DIR="/shared/results/common/miksa/intact/SD/fulleval/combo3_lr5e-06_ep3_lam0.5_nudenet_only/generated/${MODEL_NAME}"

COCO_CSV="prompts/coco_30k.csv"
COCO_REF_DIR="/shared/results/common/miksa/intact/SD/data/coco_val2014_30k_ref"
EVAL_OUTPUT_DIR="/shared/results/common/miksa/intact/SD/fulleval/combo3_lr5e-06_ep3_lam0.5_nudenet_only/eval_coco30k_probe5"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$I2P_DIR" ]; then
    echo "ERROR: I2P image directory not found: $I2P_DIR"
    exit 1
fi

TMPCONFIG="/tmp/sd_nsfw_eval_coco30k_probe5_${SLURM_JOB_ID}.yaml"

python - <<PYEOF
import os
import yaml

with open("configs/pipeline_nsfw_eval_coco30k.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["wandb"]["tags"].append("eval-only")
cfg["wandb"]["tags"].append("coco30k")
cfg["wandb"]["tags"].append("probe5")
cfg["wandb"]["group"] = "nsfw-coco30k-eval-probe5"

cfg["paths"]["output_dir"] = "${EVAL_OUTPUT_DIR}"
cfg["paths"]["coco_images_dir"] = "${COCO_REF_DIR}"
cfg["pipeline"]["model_name"] = "${MODEL_NAME}"

cfg["evaluate"]["skip_i2p"] = True
cfg["evaluate"]["probe"]["enabled"] = True
cfg["evaluate"]["n_probe_samples"] = 5
cfg["evaluate"]["coco"]["enabled"] = True
cfg["evaluate"]["coco"]["n_captions"] = 30000
cfg["evaluate"]["coco"]["pregenerated_prompts_csv"] = "${COCO_CSV}"

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py \
    --config "${TMPCONFIG}" \
    --eval-only \
    --pregenerated-images "${I2P_DIR}"

echo "SD NSFW Eval-Only COCO30k + probe5 – complete."
