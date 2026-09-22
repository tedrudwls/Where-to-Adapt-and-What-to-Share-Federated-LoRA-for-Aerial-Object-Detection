#!/usr/bin/env bash
# Four-job sequential queue for one condition/paired seed.
# Usage: bash scripts/run_heterogeneity_queue.sh <alpha0.1|iid> <42|43|44>

set -Eeuo pipefail

CONDITION="${1:-}"
SEED="${2:-}"
case "$CONDITION" in
    # Start only one Full-FT process per physical GPU when the alpha and IID
    # queues for the same seed are launched together.
    alpha0.1) METHODS=(full_ft lora fedsa_lora fixed_share_b_lora) ;;
    iid) METHODS=(fixed_share_b_lora fedsa_lora lora full_ft) ;;
    *)
        echo "Usage: $0 <alpha0.1|iid> <42|43|44>" >&2
        exit 2
        ;;
esac
case "$SEED" in
    42|43|44) ;;
    *) echo "[ERROR] Publication replicate seed must be 42, 43, or 44" >&2; exit 2 ;;
esac

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
for method in "${METHODS[@]}"; do
    echo "[QUEUE] condition=${CONDITION} seed=${SEED} method=${method}"
    CHECK_ONLY=0 bash scripts/run_heterogeneity_cell.sh "$CONDITION" "$method" "$SEED"
done
echo "[PASS] Four publication-valid cells: condition=${CONDITION}, seed=${SEED}"
