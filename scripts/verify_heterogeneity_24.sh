#!/usr/bin/env bash
# Read-only exact-protocol gate for all 24 alpha=0.1 and IID FL results.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the extracted AOD-4 Images directory}"
export SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
export RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_WEIGHTS="${MODEL_WEIGHTS:?Set MODEL_WEIGHTS to the verified pretrained RT-DETR weight}"
export DEVICE="${DEVICE:-cuda}"
cd "$PROJECT_DIR"

checked=0
for seed in 42 43 44; do
    for condition in alpha0.1 iid; do
        for method in full_ft lora fedsa_lora fixed_share_b_lora; do
            CHECK_ONLY=1 bash scripts/run_heterogeneity_cell.sh \
                "$condition" "$method" "$seed"
            checked=$((checked + 1))
        done
    done
done
if [[ "$checked" != "24" ]]; then
    echo "[ERROR] Internal verification count is $checked, expected 24" >&2
    exit 1
fi
echo "[PASS] All 24 alpha=0.1/IID results satisfy the frozen publication gate"
