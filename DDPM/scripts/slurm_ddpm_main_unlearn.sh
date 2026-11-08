#!/bin/bash
# ============================================================================
# SLURM job – single DDPM InTAct class-wise unlearning run (CIFAR-10, forget
# airplane) through main's ablation_regions.py, i.e. the exact code path that
# produced lambda_sweep/runs/ddpm_classwise_lambda_sweep_* (origin/main f91ea29
# checked out as a worktree at $MAIN_WT).
#
# Fixed paper config (hard-coded in ablation_regions.py): two_corner, full
# interval loss, k=32, Adam lr=1e-4, 3000 steps, RL objective, no remain loss.
#
# Overrides (env):
#   LAM=0          lambda_interval (0 = protection off, plain RL on the targets)
#   SEED=0
#   TARGETS=       comma-separated exact layer names to unfreeze instead of the
#                  17 paper targets (e.g. TARGETS=mid.attn_1.v)
#   RESULTS_DIR=   default: lambda_sweep (paper targets) / single_layer (TARGETS)
#
# Usage:
#   LAM=0 sbatch scripts/slurm_ddpm_main_unlearn.sh
#   LAM=0 TARGETS=mid.attn_1.v sbatch scripts/slurm_ddpm_main_unlearn.sh
# ============================================================================
#SBATCH --job-name=ddpm-main-unl
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=rtx4090_batch
#SBATCH --exclude=dgx1,dgx2
#SBATCH --requeue
#SBATCH -o /shared/results/common/miksa/intact/DDPM/slurm_logs/main_unl.%j.out

set -euo pipefail
ts() { echo "[$(date '+%F %T')] $*"; }
ts "job start on $(hostname)"

LAM=${LAM:-0}
SEED=${SEED:-0}
TARGETS=${TARGETS:-}
MAIN_WT=${MAIN_WT:-/shared/results/common/miksa/InTAct-Unl-main}
B=/shared/results/common/miksa/intact/DDPM
if [ -n "${TARGETS}" ]; then
    RESULTS_DIR=${RESULTS_DIR:-$B/single_layer}
    EXPERIMENT=grid
else
    RESULTS_DIR=${RESULTS_DIR:-$B/lambda_sweep}
    EXPERIMENT=lambda_sweep
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /shared/results/common/miksa/envs/salun-ddpm2
ts "env activated"

# CUDA smoke test before any data loading or heavy imports.  salun-ddpm2 has
# torch 2.7.1+cu128, which needs compute capability >= 7.5 (V100 = 7.0 fails
# with "no kernel image"), and the run peaks at ~16.7 GB.  Exit 3 = unusable
# card -> requeue onto another node; anything else = hard failure.
set +e
python - <<'PYEOF'
import sys, time
t0 = time.time()
import torch
print(f"torch {torch.__version__} imported in {time.time() - t0:.1f}s", flush=True)
if not torch.cuda.is_available():
    print("SMOKE FAIL: CUDA not available", flush=True); sys.exit(1)
cap = torch.cuda.get_device_capability()
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU {torch.cuda.get_device_name()} capability {cap[0]}.{cap[1]} {mem:.1f} GiB", flush=True)
if cap < (7, 5) or mem < 20:
    print("SMOKE FAIL: need compute capability >= 7.5 and >= 20 GiB", flush=True); sys.exit(3)
try:
    a = torch.randn(1024, 1024, device="cuda")
    s = float((a @ a).sum()); torch.cuda.synchronize()
except Exception as e:
    print(f"SMOKE FAIL: CUDA matmul failed: {e}", flush=True); sys.exit(1)
print(f"SMOKE OK (matmul sum {s:.3e})", flush=True)
PYEOF
SMOKE=$?
set -e
if [ ${SMOKE} -eq 3 ]; then
    ATTEMPT=${SLURM_RESTART_COUNT:-0}
    if [ "${ATTEMPT}" -lt "${MAX_REQUEUE:-20}" ]; then
        ts "unusable GPU, requeueing (attempt $((ATTEMPT + 1)))"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    ts "giving up after ${ATTEMPT} requeues"; exit 1
elif [ ${SMOKE} -ne 0 ]; then
    ts "CUDA smoke test failed (exit ${SMOKE})"; exit ${SMOKE}
fi
ts "smoke test passed (torch imported)"

cd ${MAIN_WT}/DDPM
export PYTHONPATH=${PYTHONPATH:-}:${MAIN_WT}/
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"

CONFIG=configs/pipeline_fulleval.yaml
if [ -n "${TARGETS}" ]; then
    CONFIG=/tmp/ddpm_main_unl_${SLURM_JOB_ID}.yaml
    python - <<PYEOF
import yaml
cfg = yaml.safe_load(open("configs/pipeline_fulleval.yaml"))
cfg["intact"]["targets"] = "${TARGETS}".split(",")
yaml.dump(cfg, open("${CONFIG}", "w"), default_flow_style=False)
print("targets ->", cfg["intact"]["targets"])
PYEOF
fi

ts "lambda=${LAM} seed=${SEED} targets=${TARGETS:-paper(17)} results=${RESULTS_DIR}"
python ablation_regions.py \
    --experiment ${EXPERIMENT} \
    --region_mode two_corner \
    --interval_mode full \
    --lambda ${LAM} \
    --seed ${SEED} \
    --config ${CONFIG} \
    --n_iters 3000 \
    --results_dir ${RESULTS_DIR}

ts "ablation_regions.py finished"
