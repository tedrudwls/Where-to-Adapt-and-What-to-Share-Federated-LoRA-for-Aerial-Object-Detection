#!/usr/bin/env bash
# Paired FL method comparison across Dirichlet label-skew severity.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-full_ft lora fedsa_lora fixed_share_b_lora}"
ALPHAS="${ALPHAS:-0.1 0.4 0.5 1.0}"
FIXED_PARTITION_SEED="${FIXED_PARTITION_SEED:-match}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"
MODEL_NAME="${MODEL_NAME:-rtdetr-l}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
DEVICE="${DEVICE:-cuda}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

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

alpha_tag() {
    case "$1" in
        1.0) printf '1' ;;
        *) printf '%s' "$1" ;;
    esac
}

partition_seed_for() {
    if [[ "$FIXED_PARTITION_SEED" == "match" ]]; then
        printf '%s' "$1"
    else
        printf '%s' "$FIXED_PARTITION_SEED"
    fi
}

ensure_split() {
    local alpha="$1"
    local partition_seed="$2"
    local tag
    tag="$(alpha_tag "$alpha")"
    ENSURED_SPLIT_FILE="${SPLIT_DIR}/split_official_v6_dirichlet_a${tag}_c${NUM_CLIENTS}_s${partition_seed}.json"
    if [[ ! -f "$ENSURED_SPLIT_FILE" ]]; then
        if [[ "$CHECK_ONLY" == "1" ]]; then
            echo "[ERROR] Missing required split during check-only: $ENSURED_SPLIT_FILE" >&2
            exit 1
        fi
        "$PYTHON_BIN" scripts/prepare_split.py \
            --data_root "$DATA_ROOT" \
            --output_dir "$SPLIT_DIR" \
            --source_split_policy official_aod4_v6 \
            --partition dirichlet \
            --alpha "$alpha" \
            --num_clients "$NUM_CLIENTS" \
            --seed "$partition_seed"
    fi
}

run_fl() {
    local method="$1"
    local alpha="$2"
    local train_seed="$3"
    local partition_seed="$4"
    local split_file="$5"
    local exp_name
    case "$method" in
        full_ft) exp_name="fl_full_ft_a${alpha}" ;;
        lora) exp_name="fl_lora_r8_a${alpha}" ;;
        fedsa_lora) exp_name="fl_fedsa_lora_r8_a${alpha}" ;;
        fixed_share_b_lora) exp_name="fl_fixed_share_b_lora_r8_a${alpha}" ;;
        *) echo "[ERROR] Unsupported method: $method" >&2; exit 1 ;;
    esac

    local output_dir="${RESULTS_ROOT}/seed_${train_seed}"
    local result_file="${output_dir}/${exp_name}/fl_results.json"
    local -a launch_args=(
        --data_root "$DATA_ROOT"
        --split_file "$split_file"
        --output_dir "$output_dir"
        --log_dir "${LOG_ROOT}/seed_${train_seed}"
        --exp_name "$exp_name"
        --partition dirichlet
        --dirichlet_alpha "$alpha"
        --partition_seed "$partition_seed"
        --seed "$train_seed"
        --num_clients "$NUM_CLIENTS"
        --mode fl
        --fl_method "$method"
        --fl_rounds 20
        --local_epochs 5
        --lora_rank 8
        --lora_alpha 16
        --batch_size "$BATCH_SIZE"
        --img_size 640
        --num_workers "$NUM_WORKERS"
        --device "$DEVICE"
        --close_mosaic_epochs "$CLOSE_MOSAIC_EPOCHS"
        "${MODEL_ARGS[@]}"
        --patience 0
        --no-amp
        --reset_optimizer_each_round
    )
    if [[ -f "$result_file" && "$FORCE_RERUN" != "1" ]]; then
        if result_matches_cli "$result_file" "${launch_args[@]}"; then
            echo "[SKIP] Complete result: $result_file"
            return
        fi
        echo "[ERROR] Existing result is corrupt, incomplete, or from a different protocol: $result_file" >&2
        echo "        Move it aside or set FORCE_RERUN=1." >&2
        exit 1
    fi
    if [[ "$CHECK_ONLY" == "1" ]]; then
        echo "[ERROR] Missing publication result during check-only: $result_file" >&2
        exit 1
    fi

    echo ">>> alpha=${alpha}, method=${method}, train_seed=${train_seed}, partition_seed=${partition_seed}"
    "$PYTHON_BIN" main.py "${launch_args[@]}"
    if ! result_matches_cli "$result_file" "${launch_args[@]}"; then
        echo "[ERROR] Newly completed result failed the publication gate: $result_file" >&2
        exit 1
    fi
    echo "[PASS] Publication-valid result: $result_file"
}

for train_seed in $SEEDS; do
    partition_seed="$(partition_seed_for "$train_seed")"
    for alpha in $ALPHAS; do
        ensure_split "$alpha" "$partition_seed"
        split_file="$ENSURED_SPLIT_FILE"
        for method in $METHODS; do
            run_fl "$method" "$alpha" "$train_seed" "$partition_seed" "$split_file"
        done
    done
done
