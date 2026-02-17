#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW Evaluation Only (pre-trained models)
# ============================================================================
# Evaluates already-trained models: generate all 4703 prompts → eval → wandb
# Skips the unlearning step entirely.
#
# HOW TO USE:
#   Fill in MODEL_PATHS array with your trained model checkpoint paths
#   sbatch scripts/slurm_sd_nsfw_eval_only.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-eval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-1

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Pre-trained model paths (diffusers .pt files)
# ============================================================================
COMBO_NUMBERS=(19 15)
MODEL_PATHS=(
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06/diffusers-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06.pt"
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-lr_5e-06/diffusers-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-lr_5e-06.pt"
)
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}
COMBO_NUM=${COMBO_NUMBERS[$IDX]}
MODEL_PATH=${MODEL_PATHS[$IDX]}

# Extract model name from path
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_DIR")

echo "============================================"
echo "SD NSFW Eval-Only – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}"
echo "  Model: ${MODEL_NAME}"
echo "  Path: ${MODEL_PATH}"
echo "============================================"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# ---- Build eval-only config ----
TMPCONFIG="/tmp/sd_nsfw_eval_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os, sys

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Add skip_unlearn flag
cfg["pipeline"]["skip_unlearn"] = True
cfg["pipeline"]["pretrained_model_name"] = "${MODEL_NAME}"

# Tag the wandb run
cfg["wandb"]["tags"].append("eval-only")
cfg["wandb"]["tags"].append("combo${COMBO_NUM}")
cfg["wandb"]["group"] = "nsfw-eval-only"

# Unique output dir per job
suffix = f"eval_combo{${COMBO_NUM}}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run eval-only pipeline ----
# We need to modify the pipeline call to skip unlearning
python - <<PYEOF
import sys
sys.path.insert(0, ".")

# Import and patch the pipeline to skip unlearning
import pipeline as pipe_module
import yaml
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load config
with open("${TMPCONFIG}") as f:
    cfg = yaml.safe_load(f)

# Import wandb and init
import wandb
wandb.init(
    project=cfg["wandb"]["project"],
    entity=cfg["wandb"].get("entity"),
    group=cfg["wandb"].get("group"),
    tags=cfg["wandb"].get("tags", []),
    config=cfg,
)

# Merge any wandb config overrides
cfg = pipe_module.merge_wandb_config(cfg)

import os
import torch
import numpy as np
from pathlib import Path

# Set seeds
seed = cfg["pipeline"].get("seed", 42)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device_id = cfg["pipeline"].get("device", "0")
device_str = f"cuda:{device_id}"
setting = cfg["pipeline"]["setting"]
eval_cfg = cfg.get("evaluate", {})
metrics = {}

# =========================================================================
# Skip Step 1: Unlearning (use pre-trained model)
# =========================================================================
model_name = cfg["pipeline"]["pretrained_model_name"]
log.info(f"=== Step 1: SKIPPED (using pre-trained model: {model_name}) ===")

# =========================================================================
# Step 2: Generate images
# =========================================================================
log.info("=== Step 2: Generating images ===")
images_dir = pipe_module.generate_images(cfg, model_name, device_str)
log.info(f"Images saved to {images_dir}")

# =========================================================================
# Step 3: Evaluate
# =========================================================================
log.info("=== Step 3: Evaluation ===")

image_size = cfg["unlearn"].get("image_size", 512)

# --- Probe images for NSFW ---
log.info("Generating probe images (nude + clothed prompts) for BOTH models …")
probe_dir, original_probe_dir = pipe_module.generate_nsfw_probe_images(
    model_name, cfg["paths"].get("output_dir", "./evaluation"),
    eval_cfg, device_str, cfg,
)

# --- UA (NudeNet) ---
nudenet_thresh = eval_cfg.get("nudenet", {}).get("threshold", 0.0)
ua, nn_extra = pipe_module.compute_ua_nsfw(images_dir, threshold=nudenet_thresh)
if ua is not None:
    metrics["UA"] = ua
    log.info(f"  UA (NSFW) = {ua:.4f}")
metrics.update(nn_extra)

# --- FID (clothed prompt images vs NOT_NSFW reference) ---
if eval_cfg.get("fid", {}).get("enabled", True):
    fid_cfg = eval_cfg.get("fid", {})
    max_real = fid_cfg.get("max_real", None)
    max_fake = fid_cfg.get("max_fake", None)
    not_nsfw_path = cfg["paths"].get("not_nsfw_data", "data/not-nsfw")
    fid_score = pipe_module.compute_fid_nsfw(
        probe_dir, not_nsfw_path, image_size,
        max_real=max_real, max_fake=max_fake,
    )
    if fid_score is not None:
        metrics["FID"] = fid_score
        log.info(f"  FID (clothed) = {fid_score:.2f}")

# =========================================================================
# Step 4: Log to wandb
# =========================================================================
log.info("=== Step 4: Logging ===")
wandb.log(metrics)
wandb.summary.update(metrics)

# Sample images
n_sample_imgs = eval_cfg.get("n_sample_images_per_class", 4)
pipe_module.log_sample_images_per_class(
    images_dir, setting,
    class_to_forget=None,
    n_per_class=n_sample_imgs,
    probe_dir=probe_dir,
    original_probe_dir=original_probe_dir,
)

wandb.finish()
log.info(f"Eval-only pipeline complete. Metrics: {metrics}")
PYEOF

echo "SD NSFW Eval-Only – Job ${IDX} complete."
