#!/usr/bin/env bash
# Read-only four-method MIA robustness audit for seed 42, 43, or 44.
# Usage:
#   bash scripts/run_mia_robustness_audit.sh dry-run
#   bash scripts/run_mia_robustness_audit.sh run
#   AUDIT_RESUME=1 bash scripts/run_mia_robustness_audit.sh run

set -Eeuo pipefail
umask 077

ACTION="${1:-run}"
if [[ "$ACTION" != "run" && "$ACTION" != "dry-run" ]]; then
    echo "Usage: $0 [dry-run|run]" >&2
    exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/kim/project2/data/aod4/AOD4/Images}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
AUDIT_SEED="${AUDIT_SEED:-42}"
if [[ "$AUDIT_SEED" != "42" && "$AUDIT_SEED" != "43" && "$AUDIT_SEED" != "44" ]]; then
    echo "[ERROR] AUDIT_SEED must be 42, 43, or 44; received: $AUDIT_SEED" >&2
    exit 2
fi
SPLIT_FILE="${SPLIT_FILE:-${PROJECT_DIR}/data/splits/split_official_v6_dirichlet_a0.4_c3_s${AUDIT_SEED}.json}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${PROJECT_DIR}/rtdetr-l.pt}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python3}"
if [[ "$AUDIT_SEED" == "42" ]]; then
    AUDIT_INSTANCE="security_audit_v1"
else
    AUDIT_INSTANCE="security_audit_v1_seed${AUDIT_SEED}"
fi
OUTPUT_DIR="${RESULTS_ROOT}/${AUDIT_INSTANCE}"
AUDIT_RESUME="${AUDIT_RESUME:-0}"
DEVICE="${DEVICE:-cuda}"
ATTACK_REPEATS=20
SELECTION_SEED=420042
ATTACK_SEED=842042

cd "$PROJECT_DIR"

COMMON_ARGS=(
    --project_dir "$PROJECT_DIR"
    --data_root "$DATA_ROOT"
    --split_file "$SPLIT_FILE"
    --results_root "$RESULTS_ROOT"
    --output_dir "$OUTPUT_DIR"
    --model_weights "$MODEL_WEIGHTS"
    --seed "$AUDIT_SEED"
    --methods full_ft lora fedsa_lora fixed_share_b_lora
    --device "$DEVICE"
    --selection_seed "$SELECTION_SEED"
    --attack_seed "$ATTACK_SEED"
    --attack_repeats "$ATTACK_REPEATS"
    --calibration_fraction 0.5
    --max_member_samples 1000
    --max_local_nonmember_samples 1000
    --max_pooled_nonmember_samples 2000
)

if [[ "$ACTION" == "dry-run" ]]; then
    if [[ "$AUDIT_RESUME" == "1" ]]; then
        COMMON_ARGS+=(--resume)
    fi
    exec env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
        scripts/mia_robustness_audit.py "${COMMON_ARGS[@]}" --dry_run
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    if [[ "$AUDIT_RESUME" != "1" ]]; then
        echo "[ERROR] Audit output exists: $OUTPUT_DIR" >&2
        echo "        Preserve it. Set AUDIT_RESUME=1 only for an interrupted audit." >&2
        exit 1
    fi
    if [[ -e "$OUTPUT_DIR/audit_report.json" ]]; then
        echo "[ERROR] Final audit_report.json already exists; completed/corrupt audits are immutable:" >&2
        echo "        $OUTPUT_DIR/audit_report.json" >&2
        exit 1
    fi
elif [[ "$AUDIT_RESUME" == "1" ]]; then
    echo "[ERROR] AUDIT_RESUME=1 requires an existing interrupted audit directory:" >&2
    echo "        $OUTPUT_DIR" >&2
    exit 1
fi

# The wrapper creates only the dedicated audit directory so tee cannot race the
# Python output-existence guard.  --resume is semantically safe for this empty
# first-run directory because every cache is still validated by its complete key.
mkdir -p "$OUTPUT_DIR"
chmod 700 "$OUTPUT_DIR"

CONSOLE_LOG="$(mktemp "${OUTPUT_DIR}/console.XXXXXX.log")"
chmod 600 "$CONSOLE_LOG"

set +e
env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
    scripts/mia_robustness_audit.py "${COMMON_ARGS[@]}" --resume \
    2>&1 | tee "$CONSOLE_LOG"
status="${PIPESTATUS[0]}"
set -e
chmod 600 "$CONSOLE_LOG"
echo "[Console] $CONSOLE_LOG"
exit "$status"
