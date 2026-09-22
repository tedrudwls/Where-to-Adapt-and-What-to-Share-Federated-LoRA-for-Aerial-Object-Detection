#!/usr/bin/env bash
# Image-level, ground-truth-matched loss MIA for every primary training mode.
# These are empirical leakage measurements, not a formal privacy guarantee.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
FIXED_PARTITION_SEED="${FIXED_PARTITION_SEED:-match}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLOSE_MOSAIC_EPOCHS="${CLOSE_MOSAIC_EPOCHS:-10}"
MODEL_NAME="${MODEL_NAME:-rtdetr-l}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
MIA_MAX_SAMPLES="${MIA_MAX_SAMPLES:-1000}"
MIA_CALIBRATION_FRACTION="${MIA_CALIBRATION_FRACTION:-0.5}"
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

run_or_resume_mia() {
    local result_file="$1"
    local exp_dir="$2"
    local exp_name="$3"
    shift 3
    local -a launch_args=("$@" --exp_name "$exp_name" --resume "$exp_dir")
    if [[ -f "$result_file" && "$FORCE_RERUN" != "1" ]] \
        && result_matches_cli "$result_file" --require-mia "${launch_args[@]}"; then
        echo "[SKIP] Existing MIA result: $result_file"
        return
    fi
    if [[ ! -f "$result_file" ]]; then
        echo "[ERROR] Missing primary result: $result_file" >&2
        echo "        First run: RUN_IID=0 RUN_ALPHA=0 RUN_RANK=0 RUN_TARGET_ABLATION=0 RUN_MIA=0 bash scripts/run_all_experiments.sh" >&2
        exit 1
    fi
    if ! result_matches_cli "$result_file" "${launch_args[@]}"; then
        echo "[ERROR] Primary result is corrupt, incomplete, or from a different protocol: $result_file" >&2
        echo "        Move it aside and rerun the corresponding primary experiment." >&2
        exit 1
    fi
    local checkpoint_found=0
    if [[ "$(basename "$result_file")" == "fl_results.json" ]]; then
        if [[ -f "${exp_dir}/weights/best_federated.pt" \
            || -f "${exp_dir}/weights/last_federated.pt" ]]; then
            checkpoint_found=1
        fi
    elif [[ -f "${exp_dir}/weights/best_full.pt" \
        || -f "${exp_dir}/weights/last_full.pt" ]]; then
        checkpoint_found=1
    fi
    if [[ "$checkpoint_found" != "1" ]]; then
        echo "[ERROR] Primary checkpoint is missing under: ${exp_dir}/weights" >&2
        echo "        Re-run the corresponding primary experiment before MIA." >&2
        exit 1
    fi
    echo "[RESUME] Evaluating the primary checkpoint without retraining: $exp_dir"
    "$PYTHON_BIN" main.py "${launch_args[@]}"
}

for train_seed in $SEEDS; do
    partition_seed="$(partition_seed_for "$train_seed")"
    ensure_split "$partition_seed"
    split_file="$ENSURED_SPLIT_FILE"
    output_dir="${RESULTS_ROOT}/seed_${train_seed}"
    log_dir="${LOG_ROOT}/seed_${train_seed}"

    common_args=(
        --data_root "$DATA_ROOT"
        --split_file "$split_file"
        --output_dir "$output_dir"
        --log_dir "$log_dir"
        --partition dirichlet
        --dirichlet_alpha 0.4
        --partition_seed "$partition_seed"
        --seed "$train_seed"
        --num_clients "$NUM_CLIENTS"
        --lora_rank 8
        --lora_alpha 16
        --batch_size "$BATCH_SIZE"
        --img_size 640
        --num_workers "$NUM_WORKERS"
        --close_mosaic_epochs "$CLOSE_MOSAIC_EPOCHS"
        "${MODEL_ARGS[@]}"
        --patience 0
        --no-amp
        --run_mia
        --mia_max_samples "$MIA_MAX_SAMPLES"
        --mia_calibration_fraction "$MIA_CALIBRATION_FRACTION"
    )

    for method in full_ft lora; do
        for ((client_id=0; client_id<NUM_CLIENTS; client_id++)); do
            exp_name="solo_${method}_client${client_id}"
            exp_dir="${output_dir}/${exp_name}"
            result_file="${exp_dir}/solo_results.json"
            echo ">>> MIA Solo-${method}, client=${client_id}, seed=${train_seed}"
            run_or_resume_mia "$result_file" "$exp_dir" "$exp_name" \
                "${common_args[@]}" \
                --mode solo \
                --fl_method "$method" \
                --solo_epochs 100 \
                --client_id "$client_id"
        done
    done

    for method in full_ft lora; do
        exp_name="centralized_${method}"
        exp_dir="${output_dir}/${exp_name}"
        result_file="${exp_dir}/centralized_results.json"
        echo ">>> MIA Centralized-${method}, seed=${train_seed}"
        run_or_resume_mia "$result_file" "$exp_dir" "$exp_name" \
            "${common_args[@]}" \
            --mode centralized \
            --fl_method "$method" \
            --centralized_epochs 100
    done

    for method in full_ft lora fedsa_lora fixed_share_b_lora; do
        case "$method" in
            full_ft) exp_name="fl_full_ft_a0.4" ;;
            lora) exp_name="fl_lora_r8_a0.4" ;;
            fedsa_lora) exp_name="fl_fedsa_lora_r8_a0.4" ;;
            fixed_share_b_lora) exp_name="fl_fixed_share_b_lora_r8_a0.4" ;;
        esac
        exp_dir="${output_dir}/${exp_name}"
        result_file="${exp_dir}/fl_results.json"
        echo ">>> MIA FL-${method}, seed=${train_seed}"
        run_or_resume_mia "$result_file" "$exp_dir" "$exp_name" \
            "${common_args[@]}" \
            --mode fl \
            --fl_method "$method" \
            --fl_rounds 20 \
            --local_epochs 5 \
            --reset_optimizer_each_round
    done
done
