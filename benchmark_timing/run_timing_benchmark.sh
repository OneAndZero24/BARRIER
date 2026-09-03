#!/bin/bash
# ============================================================================
# BARRIER unlearning TIMING BENCHMARK - single-run comparison of
#   SalUn | SEMU | ESC | BARRIER/InTAct
# on CIFAR-10, same model (ESC-keys AllCNN - identical weights in all four),
# same GPU, REPEATS repeat runs per method.
#
# Everything lives under the shared store:
#   $SHARED_ROOT/timing_benchmark/{envs,work,data,results}
# New conda envs are created per method, and cleaned up afterwards.
#
# Outputs (kept after cleanup):
#   results/results.jsonl            per-phase raw rows
#   results/results.csv              flat CSV (walltime per epoch + VRAM)
#   results/results_summary.csv      per-method per-phase mean walltime
#   results/results_vram.csv         per-method peak VRAM (mean/max)
#   results/logs/                    per-run stdout
#
# Usage (cluster, inside a GPU SLURM alloc or interactively):
#   sbatch .../run_timing_benchmark.sh      # via the #SBATCH header below
#   bash    .../run_timing_benchmark.sh     # interactive node with GPU
#
# Tuning knobs (env vars):
#   REPEATS=3            number of benchmark repeats
#   BATCH_SIZE=256       batch size (same across methods)
#   FORGET_CLASS=4       class to forget (ESC cifar10 convention)
#   SHARED_ROOT=/shared/results/common/miksa
#   WORK_BASE=$SHARED_ROOT/timing_benchmark
#   BARRIER_ROOT=...</>  local checkout (default: parent of this script)
#   SEMU_SRC=            optional local copy of gmum/semu instead of git clone
#   CKPT_SRC=            optional path to cifar10_ori_allcnn.pth (no download)
#   DRY_RUN=1            print commands only, execute nothing
#   KEEP_WORK=1          skip cleanup (debugging)
# ============================================================================
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=barrier-timing
#SBATCH --output=/shared/results/common/miksa/timing_benchmark/timing_%j.out
#SBATCH --error=/shared/results/common/miksa/timing_benchmark/timing_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
SHARED_ROOT="${SHARED_ROOT:-/shared/results/common/miksa}"
WORK_BASE="${WORK_BASE:-$SHARED_ROOT/timing_benchmark}"
REPEATS="${REPEATS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FORGET_CLASS="${FORGET_CLASS:-4}"
MODEL_ARCH="allcnn"
DATASET="cifar10"
NUM_FORGET=5000            # cifar10 per-class forget set (ESC convention)

REPOS="$WORK_BASE/work/repos"
DATA="$WORK_BASE/data"
ENVS="$WORK_BASE/envs"
RESULTS="$WORK_BASE/results"
RUNDIR="$WORK_BASE/work/rundir"
LOGS="$RESULTS/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARRIER_ROOT="${BARRIER_ROOT:-$(dirname "$SCRIPT_DIR")}"
CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$WORK_BASE/cache/huggingface"
export TORCH_HOME="$WORK_BASE/cache/torch"
export XDG_CACHE_HOME="$WORK_BASE/cache/xdg"
export WANDB_DIR="$WORK_BASE/cache/wandb"
export TIMING_RESULTS_DIR="$RESULTS"

DRY_RUN_FLAG="${DRY_RUN:-0}"

log()  { echo -e "\n===== $(date '+%F %T') | $* ====="; }
die()  { echo "[FATAL] $*" >&2; exit 1; }

# execute a command, or print it in DRY_RUN mode
run() {
  if [ "$DRY_RUN_FLAG" = "1" ]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Cleanup trap: remove conda envs + work dirs; keep results/ only
# ---------------------------------------------------------------------------
cleanup() {
  log "cleanup"
  if [ "${KEEP_WORK:-0}" != "1" ]; then
    for env in salun semu esc intact; do
      conda env remove -p "$ENVS/$env" -y >/dev/null 2>&1 || true
    done
    rm -rf "$REPOS" "$RUNDIR" "$WORK_BASE/cache"
    rm -rf "$DATA"                       # cifar10 re-downloads on next run
    echo "[cleanup] removed envs, repos, data; kept:"
    ls -la "$RESULTS"
  else
    echo "[cleanup] KEEP_WORK=1 - keeping $WORK_BASE"
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Conda
# ---------------------------------------------------------------------------
[ -f "$CONDA_SH" ] || die "conda.sh not found at $CONDA_SH"
# shellcheck disable=SC1091
source "$CONDA_SH"

make_env() {
  # make_env <name> <python>
  local name="$1" py="$2"
  if [ ! -x "$ENVS/$name/bin/python" ]; then
    log "creating conda env $name (py$py)"
    run conda create -y -p "$ENVS/$name" "python=$py" >/dev/null
    run "$ENVS/$name/bin/pip" install -q --upgrade pip
  else
    log "conda env $name exists, reusing"
  fi
}

install_harness_base() {
  # torch/torchvision from the pytorch cu118 wheel index, then everything
  # else from the harness requirements (minus torch/torchvision pins).
  make_env "$1" "$2"
  local pip="$ENVS/$1/bin/pip"
  run "$pip" install -q \
    "torch==$3" "torchvision==$4" \
    --index-url https://download.pytorch.org/whl/cu118
  run "$pip" install -q datasets lmdb matplotlib numpy Pillow scikit_learn six tqdm
}

mkdir -p "$WORK_BASE" "$REPOS" "$DATA" "$RESULTS" "$LOGS" "$RUNDIR"

# ---------------------------------------------------------------------------
# 2. Repos (everything in the shared store)
# ---------------------------------------------------------------------------
log "staging harnesses in $REPOS"

# salun: this repo's Classification (Unlearn-Sparse based)
[ -d "$REPOS/salun" ] || run cp -r "$BARRIER_ROOT/Classification" "$REPOS/salun"

# semu: official gmum/semu (clone once; local override allowed)
if [ ! -d "$REPOS/semu" ]; then
  if [ -n "${SEMU_SRC:-}" ]; then
    run cp -r "$SEMU_SRC" "$REPOS/semu"
  else
    run git clone --depth 1 https://github.com/gmum/semu "$REPOS/semu"
  fi
fi

# esc: ESC paper codebase (also hosts the intact/barrier run)
[ -d "$REPOS/esc" ] || run cp -r "$BARRIER_ROOT/Classification/ESC" "$REPOS/esc"
[ -d "$REPOS/intact" ] || run cp -r "$BARRIER_ROOT/Classification/ESC" "$REPOS/intact"
# InTAct package must be importable from the intact harness repo-root
# (unlearn_intact.py resolves it by walking up from $REPOS/intact, i.e.
# <work>/InTAct).
[ -d "$WORK_BASE/work/InTAct" ] || run cp -r "$BARRIER_ROOT/InTAct" "$WORK_BASE/work/InTAct"
run cp "$SCRIPT_DIR/timing_runner.py" "$REPOS/salun/timing_runner.py"
run cp "$SCRIPT_DIR/timing_runner.py" "$REPOS/semu/timing_runner.py"

# patch the salun+semu harnesses to accept arch=allcnn with ESC-keys AllCNN
run python3 "$SCRIPT_DIR/patch_models_init.py" "$REPOS/salun/models"
run python3 "$SCRIPT_DIR/patch_models_init.py" "$REPOS/semu/models"

# ---------------------------------------------------------------------------
# 3. Conda envs (one per method, in the shared store)
# ---------------------------------------------------------------------------
log "conda environments"
install_harness_base salun  3.9 2.0.1 0.15.2          # SalUn
install_harness_base semu   3.9 2.0.1 0.15.2          # SEMU (same stack)
install_harness_base esc    3.10 2.1.0 0.16.0         # ESC (torch 2.1 / timm 0.6.7)
install_harness_base intact 3.10 2.1.0 0.16.0         # BARRIER/InTAct
run "$ENVS/esc/bin/pip" install -q scikit-learn==1.2.0 "timm==0.6.7" "numpy==1.21.2" "scipy==1.9.3" gdown
run "$ENVS/intact/bin/pip" install -q scikit-learn==1.2.0 "timm==0.6.7" "numpy==1.21.2" "scipy==1.9.3"

# ---------------------------------------------------------------------------
# 4. Shared pretrained AllCNN checkpoint (ESC released cifar10_ori_allcnn.pth)
#    Keys match the allcnn_esc.AllCNN module -> the SAME weights are used by
#    all four methods.
# ---------------------------------------------------------------------------
log "checkpoint"
CKPT_DIR="$WORK_BASE/work/checkpoints"
mkdir -p "$CKPT_DIR"
CKPT="$CKPT_DIR/cifar10_ori_allcnn.pth"
if [ ! -f "$CKPT" ]; then
  if [ -n "${CKPT_SRC:-}" ]; then
    run cp "$CKPT_SRC" "$CKPT"
  else
    log "downloading ESC checkpoints (google drive folder)"
    run "$ENVS/esc/bin/python" -m gdown --folder \
      "https://drive.google.com/drive/folders/1yzahmyaNcP9Y10PTDzGdJGfqP617vrzB" \
      -O "$(dirname "$CKPT")" || \
      die "gdown failed; place cifar10_ori_allcnn.pth at $CKPT and re-run (or set CKPT_SRC)"
    # gdown may write a subfolder - look for the file
    run find "$(dirname "$CKPT")" -name "cifar10_ori_allcnn*" -exec cp {} "$CKPT" \; 2>/dev/null || true
  fi
fi
[ "$DRY_RUN_FLAG" = "1" ] || [ -f "$CKPT" ] || die "checkpoint not found at $CKPT"
echo "[ckpt] $CKPT"

# ---------------------------------------------------------------------------
# 5. SalUn saliency mask (one-time, from the pretrained model)
# ---------------------------------------------------------------------------
log "SalUn saliency mask (50% sparsity)"
run "$ENVS/salun/bin/python" -u generate_mask.py \
  --arch "$MODEL_ARCH" --dataset "$DATASET" --data "$DATA" --input_size 32 \
  --model_path "$CKPT" --save_dir "$RUNDIR/salun_mask" \
  --num_indexes_to_replace "$NUM_FORGET" --class_to_replace "$FORGET_CLASS" \
  --unlearn_epochs 1 --unlearn_lr 0.01 --batch_size "$BATCH_SIZE" --gpu 0 \
  --save_dir "$RUNDIR/salun_mask" || die "generate_mask failed"
[ "$DRY_RUN_FLAG" = "1" ] || [ -f "$RUNDIR/salun_mask/with_0.5.pt" ] || die "SalUn mask not produced"

# ---------------------------------------------------------------------------
# 6. Timed runs (REPEATS per method) via the driver
# ---------------------------------------------------------------------------
run_method() {
  local name="$1" cwd="$2"; shift 2
  for r in $(seq 1 "$REPEATS"); do
    log "RUN repeat $r/$REPEATS : $name"
    run "$ENVS/$name/bin/python" "$SCRIPT_DIR/timing_driver.py" run \
      --name "$name" --repeat "$r" --cwd "$cwd" --out results.jsonl \
      --cmd "$ENVS/$name/bin/python" -u "$@"
  done
}

# --- SalUn: masked random-label unlearning, 1 epoch = RL epoch ------------
run_method salun "$REPOS/salun" \
  timing_runner.py --method salun \
  --arch "$MODEL_ARCH" --dataset "$DATASET" --data "$DATA" --input_size 32 \
  --num_workers 4 --model_path "$CKPT" --save_dir "$RUNDIR/salun_out" \
  --mask_path "$RUNDIR/salun_mask/with_0.5.pt" \
  --unlearn RL --unlearn_epochs 3 --unlearn_lr 0.013 --batch_size "$BATCH_SIZE" \
  --class_to_replace "$FORGET_CLASS" --num_indexes_to_replace "$NUM_FORGET" \
  --seed 2 --gpu 0

# --- SEMU: SVD-transform (setup) + own_SVD training epochs -----------------
run_method semu "$REPOS/semu" \
  timing_runner.py --method semu \
  --arch "$MODEL_ARCH" --dataset "$DATASET" --data "$DATA" --input_size 32 \
  --num_workers 4 --model_path "$CKPT" --save_dir "$RUNDIR/semu_out" \
  --unlearn own_SVD --unlearn_epochs 3 --unlearn_lr 1e-5 --batch_size "$BATCH_SIZE" \
  --class_to_replace "$FORGET_CLASS" --num_indexes_to_replace "$NUM_FORGET" \
  --seed 2 --gpu 0 \
  --use_projection_grad --explained_variance_ratio 0.95

# --- ESC: one-shot SVD erase (feature pass + SVD + projection) ------------
run_method esc "$REPOS/esc" \
  unlearn_intact.py --method ESC --exp timing_esc \
  --data_name "$DATASET" --dataset_dir "$DATA" --checkpoint_dir "$CKPT_DIR" \
  --model_name AllCNN --forget_class "$FORGET_CLASS" --batch_size "$BATCH_SIZE" --p 1.5

# --- BARRIER/InTAct: interval-protected GA epochs -------------------------
run_method intact "$REPOS/intact" \
  unlearn_intact.py --method intact --exp timing_intact \
  --data_name "$DATASET" --dataset_dir "$DATA" --checkpoint_dir "$CKPT_DIR" \
  --model_name AllCNN --forget_class "$FORGET_CLASS" --batch_size "$BATCH_SIZE" \
  --unlearn_epochs 3 --intact_lambda 100 --intact_base_method ga --lr 0.01

# ---------------------------------------------------------------------------
# 7. Aggregate CSV
# ---------------------------------------------------------------------------
log "aggregating results"
run "$ENVS/esc/bin/python" "$SCRIPT_DIR/timing_driver.py" summarize \
  --summarize "$RESULTS/results.jsonl" --csv "$RESULTS/results.csv"
echo "===== DONE ====="
echo "CSV summary: $RESULTS/results_summary.csv"
echo "VRAM:        $RESULTS/results_vram.csv"
echo "rows:        $RESULTS/results.csv"
exit 0