#!/usr/bin/env bash
# Paired primary-setting ablation: global A/local B versus global B/local A.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
SEEDS="${SEEDS:-42 43 44}"
FIXED_PARTITION_SEED="${FIXED_PARTITION_SEED:-match}"
ALPHA="${ALPHA:-0.4}"
RANK="${RANK:-8}"

cd "$PROJECT_DIR"
if [[ ! -f scripts/run_fl.sh ]]; then
    echo "[ERROR] PROJECT_DIR is not the federated project: $PROJECT_DIR" >&2
    exit 1
fi

for train_seed in $SEEDS; do
    if [[ "$FIXED_PARTITION_SEED" == "match" ]]; then
        partition_seed="$train_seed"
    else
        partition_seed="$FIXED_PARTITION_SEED"
    fi
    for method in fedsa_lora fixed_share_b_lora; do
        echo ">>> Factor sharing: method=${method}, train_seed=${train_seed}, partition_seed=${partition_seed}"
        PROJECT_DIR="$PROJECT_DIR" \
        DATA_ROOT="$DATA_ROOT" \
        SPLIT_DIR="$SPLIT_DIR" \
        RESULTS_ROOT="$RESULTS_ROOT" \
        LOG_ROOT="$LOG_ROOT" \
        TRAIN_SEED="$train_seed" \
        PARTITION_SEED="$partition_seed" \
        bash scripts/run_fl.sh "$method" "$ALPHA" "$RANK"
    done
done

echo "[DONE] Fixed factor-sharing ablation under ${RESULTS_ROOT}"
