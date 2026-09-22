#!/usr/bin/env bash
# Complete reproducible experiment suite.
# Publication workload uses paired (partition, training) replicates 42, 43 and
# 44. Within each replicate every method shares the exact same split manifest.
# Set FIXED_PARTITION_SEED=42 for a fixed-split training-randomness ablation.

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
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_IID="${RUN_IID:-1}"
RUN_ALPHA="${RUN_ALPHA:-1}"
RUN_RANK="${RUN_RANK:-1}"
RUN_TARGET_ABLATION="${RUN_TARGET_ABLATION:-1}"
RUN_MIA="${RUN_MIA:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
IID_METHODS="${IID_METHODS:-full_ft lora fedsa_lora fixed_share_b_lora}"
ALPHA_METHODS="${ALPHA_METHODS:-full_ft lora fedsa_lora fixed_share_b_lora}"

cd "$PROJECT_DIR"
mkdir -p "$SPLIT_DIR" "$RESULTS_ROOT" "$LOG_ROOT"

if [[ ! -f main.py || ! -f requirements.txt || ! -f scripts/prepare_split.py \
    || ! -f scripts/result_gate.sh ]]; then
    echo "[ERROR] PROJECT_DIR is not the federated project: $PROJECT_DIR" >&2
    exit 1
fi
source scripts/result_gate.sh
MODEL_ARGS=(--model_name "$MODEL_NAME")
if [[ -n "$MODEL_WEIGHTS" ]]; then
    MODEL_ARGS+=(--model_weights "$MODEL_WEIGHTS")
fi
for split_name in train val test; do
    annotation="${DATA_ROOT}/${split_name}/_annotations.coco.json"
    if [[ ! -f "$annotation" ]]; then
        echo "[ERROR] Missing AOD-4 annotation: $annotation" >&2
        exit 1
    fi
done

partition_seed_for() {
    if [[ "$FIXED_PARTITION_SEED" == "match" ]]; then
        printf '%s' "$1"
    else
        printf '%s' "$FIXED_PARTITION_SEED"
    fi
}

ensure_base_split() {
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

run_experiment() {
    local result_file="$1"
    shift
    if [[ -f "$result_file" && "$FORCE_RERUN" != "1" ]]; then
        if result_matches_cli "$result_file" "$@"; then
            echo "[SKIP] Complete result: $result_file"
            return
        fi
        echo "[ERROR] Existing result is corrupt, incomplete, or from a different protocol: $result_file" >&2
        echo "        Move it aside or set FORCE_RERUN=1." >&2
        exit 1
    fi
    "$PYTHON_BIN" main.py "$@"
}

echo "============================================================"
echo "AOD-4 RT-DETR federated experiment suite"
echo "Project: $PROJECT_DIR"
echo "Data:    $DATA_ROOT"
echo "Seeds:   $SEEDS"
echo "Budget:  FL=20 rounds x 5 local epochs; Solo/Central=100 epochs"
echo "============================================================"

for train_seed in $SEEDS; do
    partition_seed="$(partition_seed_for "$train_seed")"
    ensure_base_split "$partition_seed"
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
    )

    # Local learning: both FT and LoRA are kept so collaboration and PEFT
    # effects are not conflated. FedSA has no distinct aggregation in solo mode.
    for method in full_ft lora; do
        for ((client_id=0; client_id<NUM_CLIENTS; client_id++)); do
            exp_name="solo_${method}_client${client_id}"
            result_file="${output_dir}/${exp_name}/solo_results.json"
            echo ">>> Solo-${method}, client=${client_id}, seed=${train_seed}"
            run_experiment "$result_file" \
                "${common_args[@]}" \
                --mode solo \
                --fl_method "$method" \
                --solo_epochs 100 \
                --client_id "$client_id" \
                --exp_name "$exp_name"
        done
    done

    # Pooled centralized upper bounds.
    for method in full_ft lora; do
        exp_name="centralized_${method}"
        result_file="${output_dir}/${exp_name}/centralized_results.json"
        echo ">>> Centralized-${method}, seed=${train_seed}"
        run_experiment "$result_file" \
            "${common_args[@]}" \
            --mode centralized \
            --fl_method "$method" \
            --centralized_epochs 100 \
            --exp_name "$exp_name"
    done

    # Primary FL comparison under the same split, seed and exposure budget.
    for method in full_ft lora fedsa_lora fixed_share_b_lora; do
        case "$method" in
            full_ft) exp_name="fl_full_ft_a0.4" ;;
            lora) exp_name="fl_lora_r8_a0.4" ;;
            fedsa_lora) exp_name="fl_fedsa_lora_r8_a0.4" ;;
            fixed_share_b_lora) exp_name="fl_fixed_share_b_lora_r8_a0.4" ;;
        esac
        result_file="${output_dir}/${exp_name}/fl_results.json"
        echo ">>> FL-${method}, seed=${train_seed}"
        run_experiment "$result_file" \
            "${common_args[@]}" \
            --mode fl \
            --fl_method "$method" \
            --fl_rounds 20 \
            --local_epochs 5 \
            --reset_optimizer_each_round \
            --exp_name "$exp_name"
    done
done

# Pass the same paths and seed policy to the focused sensitivity scripts.
export PROJECT_DIR DATA_ROOT SPLIT_DIR RESULTS_ROOT LOG_ROOT PYTHON_BIN SEEDS
export FIXED_PARTITION_SEED NUM_CLIENTS BATCH_SIZE NUM_WORKERS CLOSE_MOSAIC_EPOCHS
export MODEL_NAME MODEL_WEIGHTS
export FORCE_RERUN

if [[ "$RUN_IID" == "1" ]]; then
    METHODS="$IID_METHODS" bash scripts/run_iid_sensitivity.sh
fi
if [[ "$RUN_ALPHA" == "1" ]]; then
    METHODS="$ALPHA_METHODS" bash scripts/run_alpha_sensitivity.sh
fi
if [[ "$RUN_RANK" == "1" ]]; then
    bash scripts/run_rank_sensitivity.sh
fi
if [[ "$RUN_TARGET_ABLATION" == "1" ]]; then
    # The primary run already supplies the both-target control.
    TARGETS="decoder_only backbone_only" bash scripts/run_target_ablation.sh
fi
if [[ "$RUN_MIA" == "1" ]]; then
    bash scripts/run_mia.sh
fi

if [[ "$RUN_AGGREGATE" == "1" ]]; then
    "$PYTHON_BIN" scripts/aggregate_results.py "$RESULTS_ROOT"
fi

echo "[DONE] Results root: $RESULTS_ROOT"
