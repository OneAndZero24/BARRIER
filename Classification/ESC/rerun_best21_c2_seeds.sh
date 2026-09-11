#!/bin/bash
# ============================================================================
# Rerun the best all-around MIA config (oxy8n7qx / npzwqain, config_index=2)
# on extra seeds to get independent MIA estimates.
#
#   Row (configs/intact_mia_best21_configs.json[2]):
#     intact_lambda=0.2383289430773834 intact_forget_weight=4.504257884487311
#     intact_reduced_dim=16 unlearn_epochs=5 lr=0.00020486974164851768
#
#   Reference run: npzwqain (UA=100, RA=90.31, TA=89.86, MIA=44.0) at seed 42.
#
# The unlearned-checkpoint artifact key includes --seed (unlearn_intact.py:200),
# so each seed unlearns fresh and gets an independent MIA. Logs to the same
# wandb project/entity, run names suffixed with the seed.
#
# Usage (cluster, from Classification/ESC):
#   bash rerun_best21_c2_seeds.sh
#   SEEDS="7 99" bash rerun_best21_c2_seeds.sh   # override seeds
# ============================================================================

set -euo pipefail

SEEDS="${SEEDS:-1 2}"
CONFIG_INDEX="2"
CONFIGS_JSON="configs/intact_mia_best21_configs.json"

WANDB_ENTITY="${WANDB_ENTITY:-oneandzero24}"
WANDB_PROJECT="${WANDB_PROJECT:-esc-intact-tinyimagenet}"

ESC_ENV="${ESC_ENV:-/shared/results/common/miksa/envs/ESC}"
PY="${ESC_ENV}/bin/python"
REPO_ROOT="${REPO_ROOT:-$HOME/InTAct-Unl}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ---- Cache redirects (never in $HOME) ----
export CACHE_ROOT="${CACHE_ROOT:-/shared/results/common/miksa/esc-intact/.cache}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_DATA_HOME="$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

cd "$(dirname "$0")"

for seed in $SEEDS; do
    echo "=== [$(date '+%F %T')] best21 config_index=$CONFIG_INDEX seed=$seed ==="
    "$PY" unlearn_intact.py \
        --method intact \
        --data_name tiny_imagenet \
        --model_name vit_base_patch16_224 \
        --dataset_dir /shared/sets/datasets \
        --checkpoint_dir /shared/results/common/miksa/ESC/checkpoints \
        --forget_class 4 \
        --batch_size 64 \
        --intact_targets head \
        --intact_base_method rl \
        --mia \
        --mia_configs_json "$CONFIGS_JSON" \
        --config_index "$CONFIG_INDEX" \
        --seed "$seed" \
        --wandb \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "best21-c2-seed$seed"
done