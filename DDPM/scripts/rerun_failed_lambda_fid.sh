#!/bin/bash
# ============================================================================
# Detect + resubmit failed tasks of the ddpm-fid-repro lambda sweep
# ============================================================================
# A task is "done" iff its run dir has a rows.json sidecar (written after
# pipeline.py completes).  Any of the 35 combos without one crashed or never
# ran (typical cause: the task landed on an RTX 5090 node whose sm_120
# capability is unsupported by torch 2.1 in salun-ddpm; fixed upstream by the
# GPU guard that requeues onto a 4090).
#
# Usage (safe on the login node, no python imports):
#   bash scripts/rerun_failed_lambda_fid.sh --dry-run   # only print
#   bash scripts/rerun_failed_lambda_fid.sh             # resubmit failed idx
#   RESULTS_DIR=/custom/... bash scripts/rerun_failed_lambda_fid.sh
# ============================================================================

set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/shared/results/common/miksa/intact/DDPM/lambda_sweep}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

LAMBDAS_MAIN=(0.01 0.1 1 2 5 10 100)
SEEDS_MAIN=(0 1 2)
N_MAIN=$((7 * 3))
LAMBDAS_ONE=(0.01 0.1 1 5 10 100)
LAMBDAS_FOUR=(30 50)

FAILED=()
NEED_FID=()
DONE=0

for IDX in $(seq 0 34); do
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
        [ ${SEED} -eq 3 ] && SEED=1234
    fi

    RUN=${RESULTS_DIR}/runs/lam${LAM}_seed${SEED}
    if [ -f "${RUN}/rows.json" ]; then
        DONE=$((DONE + 1))
        if ! grep -q '"fid": [0-9]' "${RUN}/rows.json"; then
            NEED_FID+=("${IDX}")
        fi
    else
        FAILED+=("${IDX}")
    fi
done

echo "============================================================"
echo "lambda sweep status  (results: ${RESULTS_DIR})"
echo "  done : ${DONE}/35  (rows.json present)"
echo "  failed/missing: ${#FAILED[@]}/35"
echo "  done but FID missing (run backfill): ${#NEED_FID[@]}"
echo "============================================================"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "failed task ids: ${FAILED[*]}"
    for IDX in "${FAILED[@]}"; do
        if [ ${IDX} -lt ${N_MAIN} ]; then
            LAM=${LAMBDAS_MAIN[$((IDX / 3))]}; SEED=${SEEDS_MAIN[$((IDX % 3))]}
        elif [ ${IDX} -lt $((N_MAIN + 6)) ]; then
            J=$((IDX - N_MAIN)); LAM=${LAMBDAS_ONE[${J}]}; SEED=1234
        else
            J=$((IDX - N_MAIN - 6)); LAM=${LAMBDAS_FOUR[$((J / 4))]}; SEED=$((J % 4))
            [ ${SEED} -eq 3 ] && SEED=1234
        fi
        echo "    ${IDX}: lambda=${LAM} seed=${SEED}"
    done
else
    echo "all tasks done."
fi

if [ ${DRY_RUN} -eq 1 ]; then
    echo "dry-run: no resubmission."
    exit 0
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    FAILED_LIST=$(IFS=,; echo "${FAILED[*]}")
    echo "resubmitting: sbatch --array=${FAILED_LIST} ${SCRIPT_DIR}/slurm_ddpm_lambda_sweep_fid.sh"
    cd "${SCRIPT_DIR}"
    sbatch --array="${FAILED_LIST}" "${SCRIPT_DIR}/slurm_ddpm_lambda_sweep_fid.sh"
fi

if [ ${#NEED_FID[@]} -gt 0 ]; then
    echo "hint: ${#NEED_FID[@]} finished runs lack FID - resubmit the backfill:"
    echo "  sbatch scripts/slurm_fid_backfill_lambda_sweep.sh"
fi