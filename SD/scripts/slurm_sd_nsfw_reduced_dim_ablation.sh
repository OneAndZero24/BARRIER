#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW reduced_dim Ablation Study (5 dims × 3 seeds = 15)
# ============================================================================
# Varies: reduced_dim in {8, 16, 32, 64, 128}, seed in {42, 1, 2}
# Fixed:  lambda=10.0, lr=5e-6, epochs=5, Adam, base_method=nsfw,
#         targets=[attn2.to_q, attn2.to_k, attn2.to_v]
#
# Outputs per job: metrics JSON via --metrics-out, wandb logging (opt)
#
# HOW TO USE:
#   sbatch scripts/slurm_sd_nsfw_reduced_dim_ablation.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-abl-dim
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-14

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export CACHE_ROOT=/shared/results/common/miksa/intact/SD/.cache
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Grid: 5 reduced_dims × 3 seeds = 15 jobs
# Mapping: IDX = dim_index * 3 + seed_index
# ============================================================================
REDUCED_DIMS=(    8         16        32        64       128      )
SEEDS=(          42         1         2                                        )

IDX=${SLURM_ARRAY_TASK_ID}

DIM_IDX=$(( IDX / 3 ))
SEED_IDX=$(( IDX % 3 ))

DIM=${REDUCED_DIMS[$DIM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "============================================"
echo "SD NSFW Ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  reduced_dim=${DIM}  seed=${SEED}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/sd_nsfw_abl_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"
METRICS_OUT="/shared/results/common/miksa/intact/SD/ablation/reduced_dim_${DIM}_seed_${SEED}/metrics.json"

mkdir -p "$(dirname "${METRICS_OUT}")"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override varying hyperparams
cfg["intact"]["reduced_dim"]    = int("${DIM}")
cfg["pipeline"]["seed"]         = int("${SEED}")

# Fix remaining hyperparams
cfg["unlearn"]["lr"]     = 5e-6
cfg["unlearn"]["epochs"] = 5
cfg["intact"]["lambda_interval"] = 10.0

# Tag the wandb run
cfg["wandb"]["tags"].append("reduced_dim_ablation")
cfg["wandb"]["group"] = "nsfw-abl-reduced-dim"

# Unique output dir per job
suffix = f"reduced_dim_{${DIM}}_seed_{${SEED}}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py \
    --config "${TMPCONFIG}" \
    --metrics-out "${METRICS_OUT}"

echo "SD NSFW Ablation – Job ${IDX} (dim=${DIM}, seed=${SEED}) complete."
