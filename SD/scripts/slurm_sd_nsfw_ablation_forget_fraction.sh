#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW Forget-Set Bounds Fraction Ablation
# ============================================================================
# Varies: bounds_forget_fraction in {0.10, 0.25, 0.50, 0.75, 1.0}
#         seed in {42, 1, 2}
# Fixed:  bounds_remain_fraction = 1.0
#         lambda=10.0, lr=5e-6, epochs=5, Adam, base_method=nsfw,
#         reduced_dim=64, targets=[attn2.to_q, attn2.to_k, attn2.to_v]
#         use_actual_bounds=true, transf. block 2 (cross-attn QKV)
#
# Total: 5 fractions x 3 seeds = 15 jobs
#
# HOW TO USE:
#   sbatch scripts/slurm_sd_nsfw_ablation_forget_fraction.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-abl-ff
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-14

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Grid: 5 fractions x 3 seeds = 15 jobs
# Mapping: IDX = frac_index * 3 + seed_index
# ============================================================================
FRACTIONS=(0.10 0.25 0.50 0.75 1.0)
SEEDS=(42 1 2)

IDX=${SLURM_ARRAY_TASK_ID}

FRAC_IDX=$(( IDX / 3 ))
SEED_IDX=$(( IDX % 3 ))

FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Friendly label for directory names
FRAC_PCT=$(printf "%.0f" "$(echo "${FRACTION} * 100" | bc)")

echo "============================================"
echo "SD NSFW Forget-Fraction Ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  bounds_forget_fraction=${FRACTION} (${FRAC_PCT}%)  seed=${SEED}"
echo "============================================"

TMPCONFIG="/tmp/sd_nsfw_abl_ff_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"
METRICS_OUT="/shared/results/common/miksa/intact/SD/ablation/forget_frac_${FRAC_PCT}_seed_${SEED}/metrics.json"

mkdir -p "$(dirname "${METRICS_OUT}")"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["intact"]["bounds_forget_fraction"]  = float("${FRACTION}")
cfg["intact"]["bounds_remain_fraction"]  = 1.0
cfg["pipeline"]["seed"]                  = int("${SEED}")

cfg["unlearn"]["lr"]     = 5e-6
cfg["unlearn"]["epochs"] = 5
cfg["intact"]["lambda_interval"] = 10.0
cfg["intact"]["reduced_dim"]     = 64
cfg["intact"]["use_actual_bounds"] = True

cfg["wandb"]["tags"].append("forget_fraction_ablation")
cfg["wandb"]["group"] = "nsfw-abl-forget-fraction"

suffix = f"forget_frac_${FRAC_PCT}_seed_${SEED}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py \
    --config "${TMPCONFIG}" \
    --metrics-out "${METRICS_OUT}"

echo "SD NSFW Forget-Fraction Ablation – Job ${IDX} (frac=${FRAC_PCT}%, seed=${SEED}) complete."