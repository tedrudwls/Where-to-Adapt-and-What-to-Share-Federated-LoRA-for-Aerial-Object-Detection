#!/usr/bin/env bash
# Run exactly one frozen alpha=0.1 or IID FL cell.
# Usage: bash scripts/run_heterogeneity_cell.sh <alpha0.1|iid> <method> <42|43|44>

set -Eeuo pipefail

CONDITION="${1:-}"
METHOD="${2:-}"
SEED="${3:-}"

case "$CONDITION" in
    alpha0.1|iid) ;;
    *)
        echo "Usage: $0 <alpha0.1|iid> <full_ft|lora|fedsa_lora|fixed_share_b_lora> <42|43|44>" >&2
        exit 2
        ;;
esac
case "$METHOD" in
    full_ft|lora|fedsa_lora|fixed_share_b_lora) ;;
    *)
        echo "[ERROR] Unsupported method: $METHOD" >&2
        exit 2
        ;;
esac
case "$SEED" in
    42|43|44) ;;
    *)
        echo "[ERROR] Publication replicate seed must be 42, 43, or 44" >&2
        exit 2
        ;;
esac

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python3}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${PROJECT_DIR}/rtdetr-l.pt}"

common_env=(
    PROJECT_DIR="$PROJECT_DIR"
    DATA_ROOT="$DATA_ROOT"
    SPLIT_DIR="$SPLIT_DIR"
    RESULTS_ROOT="$RESULTS_ROOT"
    LOG_ROOT="$LOG_ROOT"
    PYTHON_BIN="$PYTHON_BIN"
    MODEL_NAME="rtdetr-l"
    MODEL_WEIGHTS="$MODEL_WEIGHTS"
    DEVICE="cuda"
    SEEDS="$SEED"
    METHODS="$METHOD"
    FIXED_PARTITION_SEED="match"
    NUM_CLIENTS="3"
    BATCH_SIZE="8"
    NUM_WORKERS="4"
    CLOSE_MOSAIC_EPOCHS="10"
    FORCE_RERUN="0"
    CHECK_ONLY="${CHECK_ONLY:-0}"
)

cd "$PROJECT_DIR"
if [[ "$CONDITION" == "alpha0.1" ]]; then
    exec env "${common_env[@]}" ALPHAS="0.1" \
        bash scripts/run_alpha_sensitivity.sh
fi
exec env "${common_env[@]}" bash scripts/run_iid_sensitivity.sh
