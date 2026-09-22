#!/usr/bin/env bash
# Reproduce one alpha=0.4 FL LoRA placement/rank cell at a time.
# Usage: bash scripts/run_lora_grid.sh METHOD RANK TARGET SEED
# METHOD: lora | fedsa_lora | fixed_share_b_lora
# RANK/TARGET: 8 with decoder_only | backbone_only | both,
#              or 4/16 with both only.
# SEED: 42 | 43 | 44 (training and partition seeds are paired).

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_lora_grid.sh METHOD RANK TARGET SEED

  METHOD  lora | fedsa_lora | fixed_share_b_lora
  RANK    4 | 8 | 16
  TARGET  decoder_only | backbone_only | both
  SEED    42 | 43 | 44

Only rank 8 supports placement-only cells. Ranks 4 and 16 use both targets.
This runs one FL experiment (20 rounds x 5 local epochs) on the official
AOD-4 v6 split with Dirichlet alpha=0.4. Training and partition seeds match.

Environment: PROJECT_DIR, DATA_ROOT, SPLIT_DIR, RESULTS_ROOT, LOG_ROOT,
PYTHON_BIN, MODEL_WEIGHTS, CUDA_VISIBLE_DEVICES. Set DRY_RUN=1 to print the
resolved experiment without touching data, outputs, or checkpoints.

Existing complete matching results are skipped. Incomplete or mismatched
results, and nonempty experiment directories without a result, are never
overwritten; move them aside explicitly before retrying.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 4 )); then
    usage >&2
    exit 2
fi

method="$1"
rank="$2"
target="$3"
seed="$4"

case "$method" in
    lora|fedsa_lora|fixed_share_b_lora) ;;
    *) echo "[ERROR] Unsupported method: $method" >&2; exit 2 ;;
esac
case "$rank" in
    4|8|16) ;;
    *) echo "[ERROR] Rank must be 4, 8, or 16" >&2; exit 2 ;;
esac
case "$target" in
    decoder_only|backbone_only|both) ;;
    *) echo "[ERROR] Unsupported target: $target" >&2; exit 2 ;;
esac
case "$seed" in
    42|43|44) ;;
    *) echo "[ERROR] Seed must be 42, 43, or 44" >&2; exit 2 ;;
esac
if [[ "$rank" != "8" && "$target" != "both" ]]; then
    echo "[ERROR] Placement-only runs were evaluated only at rank 8" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$script_dir/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$DRY_RUN" in
    0|1) ;;
    *) echo "[ERROR] DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -z "$DATA_ROOT" ]]; then
    echo "[ERROR] Set DATA_ROOT to the extracted AOD-4 Images directory" >&2
    exit 1
fi

exp_name="fl_${method}_r${rank}_a0.4"
target_args=()
case "$target" in
    decoder_only)
        exp_name="${exp_name}_decoder_only"
        target_args=(--no-apply_lora_backbone --apply_lora_decoder)
        ;;
    backbone_only)
        exp_name="${exp_name}_backbone_only"
        target_args=(--apply_lora_backbone --no-apply_lora_decoder)
        ;;
    both)
        target_args=(--apply_lora_backbone --apply_lora_decoder)
        ;;
esac

split_file="${SPLIT_DIR}/split_official_v6_dirichlet_a0.4_c3_s${seed}.json"
output_dir="${RESULTS_ROOT}/seed_${seed}"
experiment_dir="${output_dir}/${exp_name}"
result_file="${experiment_dir}/fl_results.json"
model_args=(--model_name rtdetr-l)
if [[ -n "$MODEL_WEIGHTS" ]]; then
    model_args+=(--model_weights "$MODEL_WEIGHTS")
fi

launch_args=(
    --data_root "$DATA_ROOT"
    --split_file "$split_file"
    --output_dir "$output_dir"
    --log_dir "${LOG_ROOT}/seed_${seed}"
    --exp_name "$exp_name"
    --partition dirichlet
    --dirichlet_alpha 0.4
    --partition_seed "$seed"
    --seed "$seed"
    --num_clients 3
    --mode fl
    --fl_method "$method"
    --fl_rounds 20
    --local_epochs 5
    --lora_rank "$rank"
    --lora_alpha "$((2 * rank))"
    "${target_args[@]}"
    --batch_size 8
    --img_size 640
    --num_workers 4
    --close_mosaic_epochs 10
    "${model_args[@]}"
    --patience 0
    --no-amp
    --reset_optimizer_each_round
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'experiment_id=seed_%s/%s\n' "$seed" "$exp_name"
    printf 'result_file=%s\n' "$result_file"
    printf 'split_file=%s\n' "$split_file"
    printf 'command='
    printf '%q ' "$PYTHON_BIN" "$PROJECT_DIR/main.py" "${launch_args[@]}"
    printf '\n'
    exit 0
fi

if [[ -z "$MODEL_WEIGHTS" || ! -f "$MODEL_WEIGHTS" ]]; then
    echo "[ERROR] Set MODEL_WEIGHTS to the verified pretrained RT-DETR checkpoint" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/main.py" || ! -f "${PROJECT_DIR}/scripts/result_gate.sh" ||
      ! -f "${PROJECT_DIR}/scripts/prepare_split.py" ]]; then
    echo "[ERROR] PROJECT_DIR is not the federated project: $PROJECT_DIR" >&2
    exit 1
fi
cd "$PROJECT_DIR"
source scripts/result_gate.sh

if [[ -f "$result_file" ]]; then
    if result_matches_cli "$result_file" "${launch_args[@]}"; then
        echo "[SKIP] Complete matching result: $result_file"
        exit 0
    fi
    echo "[ERROR] Existing result is incomplete or protocol-mismatched: $result_file" >&2
    echo "        Move the experiment directory aside explicitly before retrying." >&2
    exit 1
fi
if [[ -d "$experiment_dir" ]] &&
   [[ -n "$(find "$experiment_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[ERROR] Existing nonempty experiment directory: $experiment_dir" >&2
    echo "        Inspect and move it aside explicitly before retrying." >&2
    exit 1
fi

if [[ ! -f "$split_file" ]]; then
    "$PYTHON_BIN" scripts/prepare_split.py \
        --data_root "$DATA_ROOT" \
        --output_dir "$SPLIT_DIR" \
        --source_split_policy official_aod4_v6 \
        --partition dirichlet \
        --alpha 0.4 \
        --num_clients 3 \
        --seed "$seed"
fi

echo "[RUN] seed=${seed}, method=${method}, rank=${rank}, target=${target}"
"$PYTHON_BIN" main.py "${launch_args[@]}"
