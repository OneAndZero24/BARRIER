#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM InTAct lambda_interval sweep on the ORIGINAL FID setup
# (branch ddpm-fid-repro, "Initial public release" pipeline)
# ============================================================================
# Repeats the full lambda-front experiment (main sweep + extra pareto points)
# exactly like scripts/slurm_ddpm_lambda_sweep.sh / _extra.sh on master, but
# driven through the ORIGINAL pipeline.py + TF-Based Inception-V3 FID
# (evaluator.py, pool_3 2048-dim, 5000 samples/class) instead of the newer
# ablation_regions.py / fid_only.py code path.
#
# Swept (35 jobs, array 0-34):
#   A) main sweep : lambda_interval {0.01, 0.1, 1, 2, 5, 10, 100} x seeds {0,1,2}
#   B) extra      : lambda {0.01, 0.1, 1, 5, 10, 100} at canonical seed 1234
#                   lambda {30, 50} x seeds {0, 1, 2, 1234}
#
# Fixed paper configuration (same as master lambda sweeps):
#   targets = QKV attention projections + class-embedding MLP (11 targets),
#   reduced_dim k=32, Adam, lr=1e-4, 3000 steps, RL objective, two_corner /
#   use_actual_bounds=true (full interval loss), no remain-set loss (alpha 0).
#
# Each job patches configs/pipeline_fulleval.yaml (lambda, seed, lr, n_iters,
# per-run output dirs), runs pipeline.py (unlearn -> sample -> FID/UA/TA),
# and writes the per-run sidecar <results_dir>/runs/lamX_seedY/rows.json
# (keys: lambda, seed, ua, ta, fid) for offline aggregation.
#
# Env: salun-ddpm (original paper env) — already matches DDPM/requirements.txt
# (tensorflow==2.12.0 for the FID graph, numpy==1.23.5, ...; torch 2.1.0 vs
# pinned 2.0.1, which is what the original runs actually used).  To redeploy
# from scratch instead:
#   conda create -n salun-ddpm2 python=3.8 pip=23.1.2
#   conda activate salun-ddpm2 && pip install -r requirements.txt wandb
#
# After the array completes, aggregate + render the pareto curves:
#   python scripts/plot_lambda_sweep.py \
#       --results_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep
#
# Re-run any tasks that failed (e.g. landed on an RTX 5090 node whose sm_120
# is unsupported by torch 2.1):
#   bash scripts/rerun_failed_lambda_fid.sh          # prints + resubmits
#   bash scripts/rerun_failed_lambda_fid.sh --dry-run   # only prints
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_lambda_sweep_fid.sh
# ============================================================================

#SBATCH --job-name=ddpm-lambda-fid-repro
#SBATCH --output=slurm-%A_%a.out
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-34
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail

# ============================================================================
# GPU guard: rtx4090_batch occasionally schedules RTX 5090 (sm_120/Blackwell)
# nodes, but torch 2.1.0 in salun-ddpm has no sm_120 kernels ("no kernel image
# is available for execution on the device").  Detect the GPU first and
# requeue until the job lands on a pre-Blackwell node (4090, A100, ...).
# SLURM_RESTART_COUNT increments on each requeue; MAX_REQUEUE caps the loop.
# If the partition has usable node features you can skip the loop entirely:
#   sinfo -p rtx4090_batch -N -o "%N %f"       # list node features
#   #SBATCH --constraint=rtx4090
# ============================================================================
MAX_REQUEUE=${MAX_REQUEUE:-20}
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: ${GPU_NAME}  (compute capability ${GPU_CAP:-unknown})"
if [ -n "${GPU_CAP}" ] && [ "${GPU_CAP}" -ge 10 ]; then
    ATTEMPT=${SLURM_RESTART_COUNT:-0}
    echo "FATAL-REQUEUE: capability ${GPU_CAP}.x (Blackwell sm_120+) unsupported by torch 2.1 kernels"
    if [ "${ATTEMPT}" -lt "${MAX_REQUEUE}" ]; then
        echo "requeueing (attempt $((ATTEMPT + 1))/${MAX_REQUEUE}) ..."
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    echo "giving up after ${MAX_REQUEUE} requeues - no usable GPU found"
    exit 1
fi

CONDA_ENV=${CONDA_ENV:-salun-ddpm}
RESULTS_DIR=${RESULTS_DIR:-/shared/results/common/miksa/intact/DDPM/lambda_sweep}
WANDB_MODE=${WANDB_MODE:-offline}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=${PYTHONPATH:-}:/home/miksa/InTAct-Unl/

export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"
export WANDB_MODE="${WANDB_MODE}"

# ---- Env sanity check (TF needed for the old-commit FID evaluator) ----
python - <<'CHECKEOF'
try:
    import tensorflow, torch, numpy
    import tensorflow.compat.v1 as tf
    print(f"tf={tensorflow.__version__} torch={torch.__version__} numpy={numpy.__version__}")
    tf.disable_v2_behavior()
except Exception as e:
    raise SystemExit(f"env check failed (needs TF for FID): {e}")
CHECKEOF

# ============================================================================
# Sweep table
# ============================================================================
LAMBDAS_MAIN=(0.01 0.1 1 2 5 10 100)
SEEDS_MAIN=(0 1 2)
N_MAIN=$((7 * 3))                      # 21  -> array 0-20
LAMBDAS_ONE=(0.01 0.1 1 5 10 100)      # seed 1234 reruns -> array 21-26
LAMBDAS_FOUR=(30 50)                   # all 4 seeds       -> array 27-34

IDX=${SLURM_ARRAY_TASK_ID}

if [ ${IDX} -lt ${N_MAIN} ]; then
    LAM=${LAMBDAS_MAIN[$((IDX / 3))]}
    SEED=${SEEDS_MAIN[$((IDX % 3))]}
elif [ ${IDX} -lt $((N_MAIN + 6)) ]; then
    J=$((IDX - N_MAIN))
    LAM=${LAMBDAS_ONE[${J}]}
    SEED=1234
else
    J=$((IDX - N_MAIN - 6))
    LAM=${LAMBDAS_FOUR[$((J / 4))]}
    SEED=$((J % 4))
    case ${SEED} in 3) SEED=1234;; esac
fi

echo "============================================"
echo "lambda sweep (old FID pipeline) – ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  lambda_interval=${LAM}  seed=${SEED}"
echo "============================================"

RUN_DIR=${RESULTS_DIR}/runs/lam${LAM}_seed${SEED}
TMPCONFIG="/tmp/ddpm_lambda_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

# ---- Build per-job config by patching the full-eval template ----
python - <<PYEOF
import os, yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["pipeline"]["seed"] = int("${SEED}")
cfg["unlearn"]["lr"] = 1e-4
cfg["unlearn"]["n_iters"] = 3000
cfg["unlearn"]["method"] = "rl"
cfg["unlearn"]["label_to_forget"] = 0
cfg["intact"]["lambda_interval"] = float("${LAM}")
cfg["intact"]["use_actual_bounds"] = True
cfg["wandb"]["group"] = "lambda-sweep-fid"
cfg["wandb"]["tags"] = ["ddpm", "cifar10", "intact", "fulleval",
                        "lambda_sweep", "fid-repro"]
cfg["paths"]["output_dir"] = os.path.join("${RUN_DIR}", "output")
cfg["paths"]["checkpoint_dir"] = os.path.join("${RUN_DIR}", "output", "ckpts")

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print("config written to ${TMPCONFIG}")
PYEOF

# ---- Run the ORIGINAL pipeline (unlearn -> sample -> TF FID / UA / TA) ----
python pipeline.py --config "${TMPCONFIG}"

# ---- Sidecar for offline aggregation (no wandb required) ----
python scripts/write_lambda_sidecar.py \
    --run_dir "${RUN_DIR}" \
    --lambda "${LAM}" \
    --seed "${SEED}"

echo "lambda sweep – index ${IDX} (lambda=${LAM}, seed=${SEED}) complete."