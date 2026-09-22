#!/usr/bin/env bash
# CPU-only, read-only phase-2 audit over the frozen v1 per-image loss caches.
# Usage: bash scripts/run_mia_covariate_audit.sh [dry-run|run]

set -Eeuo pipefail
umask 077

ACTION="${1:-run}"
if [[ "$ACTION" != "run" && "$ACTION" != "dry-run" ]]; then
    echo "Usage: $0 [dry-run|run]" >&2
    exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/gpuadmin/kim/fedsalora}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/official_v6}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python3}"
OUTPUT_DIR="${RESULTS_ROOT}/security_audit_v2_covariate_multiseed"

cd "$PROJECT_DIR"
if [[ ! -f scripts/mia_covariate_audit.py ]]; then
    echo "[ERROR] Missing scripts/mia_covariate_audit.py" >&2
    exit 1
fi

args=(
    --results_root "$RESULTS_ROOT"
    --output_dir "$OUTPUT_DIR"
)
if [[ "$ACTION" == "dry-run" ]]; then
    args+=(--dry_run)
fi

exec env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
    scripts/mia_covariate_audit.py "${args[@]}"
