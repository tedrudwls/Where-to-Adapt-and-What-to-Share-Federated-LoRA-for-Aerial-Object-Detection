#!/usr/bin/env bash
# Usage: bash scripts/run_solo.sh [CLIENT_ID|all] [full_ft|lora]

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
PARTITION_SEED="${PARTITION_SEED:-42}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"
MODEL_NAME="${MODEL_NAME:-rtdetr-l}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CLIENT_ID="${1:-all}"
METHOD="${2:-lora}"
SPLIT_FILE="${SPLIT_DIR}/split_official_v6_dirichlet_a0.4_c${NUM_CLIENTS}_s${PARTITION_SEED}.json"
OUTPUT_DIR="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}/seed_${TRAIN_SEED}"
LOG_DIR="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}/seed_${TRAIN_SEED}"

cd "$PROJECT_DIR"
if [[ ! -f main.py || ! -f scripts/prepare_split.py || ! -f scripts/result_gate.sh ]]; then
    echo "[ERROR] PROJECT_DIR is not the federated project: $PROJECT_DIR" >&2
    exit 1
fi
source scripts/result_gate.sh
MODEL_ARGS=(--model_name "$MODEL_NAME")
if [[ -n "$MODEL_WEIGHTS" ]]; then
    MODEL_ARGS+=(--model_weights "$MODEL_WEIGHTS")
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
    "$PYTHON_BIN" scripts/prepare_split.py \
        --data_root "$DATA_ROOT" --output_dir "$SPLIT_DIR" \
        --source_split_policy official_aod4_v6 \
        --partition dirichlet --alpha 0.4 --num_clients "$NUM_CLIENTS" --seed "$PARTITION_SEED"
fi
if [[ "$METHOD" != "full_ft" && "$METHOD" != "lora" ]]; then
    echo "[ERROR] METHOD must be full_ft or lora" >&2
    exit 1
fi

run_client() {
    local client_id="$1"
    local exp_name="solo_${METHOD}_client${client_id}"
    local -a launch_args=(
        --data_root "$DATA_ROOT" --split_file "$SPLIT_FILE" \
        --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR" --exp_name "$exp_name" \
        --partition dirichlet --dirichlet_alpha 0.4 \
        --partition_seed "$PARTITION_SEED" --seed "$TRAIN_SEED" \
        --num_clients "$NUM_CLIENTS" \
        --mode solo --client_id "$client_id" --fl_method "$METHOD" \
        --solo_epochs 100 --lora_rank 8 --lora_alpha 16 \
        --batch_size "$BATCH_SIZE" --img_size 640 --num_workers "$NUM_WORKERS" \
        --close_mosaic_epochs "$CLOSE_MOSAIC_EPOCHS" \
        "${MODEL_ARGS[@]}" \
        --patience 0 --no-amp
    )
    local result_file="${OUTPUT_DIR}/${exp_name}/solo_results.json"
    if [[ -f "$result_file" && "$FORCE_RERUN" != "1" ]]; then
        if result_matches_cli "$result_file" "${launch_args[@]}"; then
            echo "[SKIP] Complete matching result: $result_file"
            return
        fi
        echo "[ERROR] Existing result is corrupt, incomplete, or from a different protocol: $result_file" >&2
        echo "        Move it aside or set FORCE_RERUN=1." >&2
        exit 1
    fi
    "$PYTHON_BIN" main.py "${launch_args[@]}"
}

if [[ "$CLIENT_ID" == "all" ]]; then
    for ((client_id=0; client_id<NUM_CLIENTS; client_id++)); do
        run_client "$client_id"
    done
else
    run_client "$CLIENT_ID"
fi
