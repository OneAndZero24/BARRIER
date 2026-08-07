#!/bin/bash -l
# ============================================================================
# SLURM – DDPM CIFAR-10 InTAct Ablation: SVD vs PCA vs Raw activation space
# ============================================================================
# Compares four variants of the InTAct protection:
#
#   Combo 0 : Standard InTAct (SVD + percentile intervals + U_residual)
#   Combo 1 : No SVD — protect on RAW activation space (per-dim quantiles)
#   Combo 2 : SVD without intervals — keep SVD, switch off U_forget drift
#             (only U_residual + mu protection remain)
#   Combo 3 : PCA (eigendecomposition via eigh) instead of SVD, same
#             interval structure
#
# Fixed: forget_class=0, lr=1e-4, n_iters=3000, lambda_interval=5.0,
#        method=rl, targets=QKV+cemb, reduced_dim=32
#
# Outputs: wandb metrics (FID,UA/FA/TA), model checkpoints
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_ablation_svd_pca_raw.sh
# ============================================================================

#SBATCH --job-name=ddpm-svd-abl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=dgxh100
#SBATCH --qos=big
#SBATCH --array=0-4

set -euo pipefail

# ---- Environment ----
source /home/miksa/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd "$HOME/InTAct-Unl/DDPM"
export PYTHONPATH="$HOME/InTAct-Unl/taming-transformers:$HOME/InTAct-Unl:${PYTHONPATH:-}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-/shared/results/common/miksa/.cache/huggingface/token}"
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
    HUGGINGFACE_HUB_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

CACHE_BASE="/shared/results/common/miksa/.cache"
export CACHE_ROOT="$CACHE_BASE"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

RESULTS_BASE="/shared/results/common/miksa/intact/DDPM/ablation_svd_pca_raw"
mkdir -p "$RESULTS_BASE"

cleanup() {
    local rc=$?
    echo "=== CLEANUP (exit code $rc) ==="
    rm -f "${TMPCONFIG:-/dev/null}" 2>/dev/null || true
    exit $rc
}
trap cleanup EXIT

IDX=${SLURM_ARRAY_TASK_ID}

# ============================================================================
# Grid: 5 ablation variants
# ============================================================================
SKIP_SVD=(           "false"   "true"    "false"   "false"   "false")
SKIP_INTERVAL=(      "false"   "false"   "true"    "false"   "false")
REMOVE_TOP_DIRS=(    "false"   "false"   "false"   "false"   "true")
DECOMP_METHOD=(      "svd"     "svd"     "svd"     "pca"     "svd")
VARIANT_LABELS=(
    "standard_svd"
    "no_svd_raw"
    "svd_no_interval"
    "pca_eigh"
    "remove_top_dirs"
)

SKIP_SVD_VAL=${SKIP_SVD[$IDX]}
SKIP_INTERVAL_VAL=${SKIP_INTERVAL[$IDX]}
REMOVE_TOP_VAL=${REMOVE_TOP_DIRS[$IDX]}
DECOMP_VAL=${DECOMP_METHOD[$IDX]}
VARIANT=${VARIANT_LABELS[$IDX]}

echo "============================================"
echo "DDPM SVD/PCA/Raw Ablation on $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID:-local}"
echo "  Array:    ${IDX}"
echo "  Variant:  ${VARIANT}"
echo "    skip_svd            = ${SKIP_SVD_VAL}"
echo "    skip_interval       = ${SKIP_INTERVAL_VAL}"
echo "    remove_top_dirs     = ${REMOVE_TOP_VAL}"
echo "    decomp_method       = ${DECOMP_VAL}"
echo "============================================"

# ============================================================================
# Build per-job config
# ============================================================================
TMPCONFIG="/tmp/ddpm_svd_abl_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import os, yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Fixed hyperparams
cfg["unlearn"]["label_to_forget"] = 0
cfg["unlearn"]["lr"] = 1e-4
cfg["unlearn"]["n_iters"] = 3000
cfg["unlearn"]["method"] = "rl"

# InTAct params
cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = 5.0
cfg["intact"]["reduced_dim"] = 32
cfg["intact"]["lower_percentile"] = 0.05
cfg["intact"]["upper_percentile"] = 0.95
cfg["intact"]["use_actual_bounds"] = False
cfg["intact"]["normalize_protection"] = True

# Variant-specific flags
cfg["intact"]["skip_svd"]              = "${SKIP_SVD_VAL}".lower() == "true"
cfg["intact"]["skip_interval"]         = "${SKIP_INTERVAL_VAL}".lower() == "true"
cfg["intact"]["remove_top_directions"] = "${REMOVE_TOP_VAL}".lower() == "true"
cfg["intact"]["decomp_method"]         = "${DECOMP_VAL}"

# Shared paths
cfg["paths"]["pretrained_ckpt_folder"] = "/shared/results/common/miksa/intact/DDPM/results/cifar10/2026_01_13_220000"
cfg["paths"]["output_dir"]      = os.path.join("${RESULTS_BASE}", "${VARIANT}", "output")
cfg["paths"]["checkpoint_dir"]  = os.path.join("${RESULTS_BASE}", "${VARIANT}", "checkpoints")
cfg["paths"]["ref_dataset_dir"] = "/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"
cfg["paths"]["classifier_ckpt"] = "/shared/results/common/miksa/intact/DDPM/models/cifar10_resnet34.pth"

# Eval budgets
cfg.setdefault("evaluate", {})
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

# Wandb tags
cfg["wandb"]["group"] = "cifar10-svd-pca-raw-ablation"
cfg["wandb"]["tags"] = [
    "ablation", "svd-pca-raw",
    f"variant-${VARIANT}",
    "skip_svd_${SKIP_SVD_VAL}",
    "skip_interval_${SKIP_INTERVAL_VAL}",
    "decomp_${DECOMP_VAL}",
    "lr_1e-4", "iters_3000", "lambda_5.0"
]

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ============================================================================
# Run pipeline
# ============================================================================
python pipeline.py --config "${TMPCONFIG}"

echo "DDPM SVD/PCA/Raw ablation – job ${IDX} (${VARIANT}) complete."
