#!/usr/bin/env bash
# FedSA-LoRA rank sensitivity at Dirichlet alpha=0.4.
# LoRA alpha is set to 2*r, so alpha/r remains exactly 2 for every rank.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
RANKS="${RANKS:-4 8 16}"
FIXED_PARTITION_SEED="${FIXED_PARTITION_SEED:-match}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"
MODEL_NAME="${MODEL_NAME:-rtdetr-l}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

cd "$PROJECT_DIR"
mkdir -p "$SPLIT_DIR" "$RESULTS_ROOT" "$LOG_ROOT"

if [[ ! -f main.py || ! -f scripts/prepare_split.py || ! -f scripts/result_gate.sh ]]; then
    echo "[ERROR] PROJECT_DIR is not the federated project: $PROJECT_DIR" >&2
    exit 1
fi
source scripts/result_gate.sh
MODEL_ARGS=(--model_name "$MODEL_NAME")
if [[ -n "$MODEL_WEIGHTS" ]]; then
    MODEL_ARGS+=(--model_weights "$MODEL_WEIGHTS")
fi

partition_seed_for() {
    if [[ "$FIXED_PARTITION_SEED" == "match" ]]; then
        printf '%s' "$1"
    else
        printf '%s' "$FIXED_PARTITION_SEED"
    fi
}

ensure_split() {
    local partition_seed="$1"
    ENSURED_SPLIT_FILE="${SPLIT_DIR}/split_official_v6_dirichlet_a0.4_c${NUM_CLIENTS}_s${partition_seed}.json"
    if [[ ! -f "$ENSURED_SPLIT_FILE" ]]; then
        "$PYTHON_BIN" scripts/prepare_split.py \
            --data_root "$DATA_ROOT" \
            --output_dir "$SPLIT_DIR" \
            --source_split_policy official_aod4_v6 \
            --partition dirichlet \
            --alpha 0.4 \
            --num_clients "$NUM_CLIENTS" \
            --seed "$partition_seed"
    fi
}

for train_seed in $SEEDS; do
    partition_seed="$(partition_seed_for "$train_seed")"
    ensure_split "$partition_seed"
    split_file="$ENSURED_SPLIT_FILE"
    output_dir="${RESULTS_ROOT}/seed_${train_seed}"
    for rank in $RANKS; do
        lora_alpha=$((2 * rank))
        exp_name="fl_fedsa_lora_r${rank}_a0.4"
        result_file="${output_dir}/${exp_name}/fl_results.json"
        launch_args=(
            --data_root "$DATA_ROOT"
            --split_file "$split_file"
            --output_dir "$output_dir"
            --log_dir "${LOG_ROOT}/seed_${train_seed}"
            --exp_name "$exp_name"
            --partition dirichlet
            --dirichlet_alpha 0.4
            --partition_seed "$partition_seed"
            --seed "$train_seed"
            --num_clients "$NUM_CLIENTS"
            --mode fl
            --fl_method fedsa_lora
            --fl_rounds 20
            --local_epochs 5
            --lora_rank "$rank"
            --lora_alpha "$lora_alpha"
            --batch_size "$BATCH_SIZE"
            --img_size 640
            --num_workers "$NUM_WORKERS"
            --close_mosaic_epochs "$CLOSE_MOSAIC_EPOCHS"
            "${MODEL_ARGS[@]}"
            --patience 0
            --no-amp
            --reset_optimizer_each_round
        )
        if [[ -f "$result_file" && "$FORCE_RERUN" != "1" ]]; then
            if result_matches_cli "$result_file" "${launch_args[@]}"; then
                echo "[SKIP] Complete result: $result_file"
                continue
            fi
            echo "[ERROR] Existing result is corrupt, incomplete, or from a different protocol: $result_file" >&2
            echo "        Move it aside or set FORCE_RERUN=1." >&2
            exit 1
        fi

        echo ">>> rank=${rank}, lora_alpha=${lora_alpha}, train_seed=${train_seed}, partition_seed=${partition_seed}"
        "$PYTHON_BIN" main.py "${launch_args[@]}"
    done
done
