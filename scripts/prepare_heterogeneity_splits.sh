#!/usr/bin/env bash
# Serially create the six immutable split manifests needed by the 24-run study.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the extracted AOD-4 Images directory}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
mkdir -p "$SPLIT_DIR"
for seed in 42 43 44; do
    alpha_split="${SPLIT_DIR}/split_official_v6_dirichlet_a0.1_c3_s${seed}.json"
    alpha_yolo="${SPLIT_DIR}/yolo_official_v6_dirichlet_a0.1_c3_s${seed}"
    if [[ -e "$alpha_split" || -e "$alpha_yolo" ]]; then
        if [[ ! -f "$alpha_split" || ! -d "$alpha_yolo" ]]; then
            echo "[ERROR] Partial alpha=0.1 split artifact pair: $alpha_split / $alpha_yolo" >&2
            exit 1
        fi
        echo "[SKIP] Existing alpha=0.1 split pair: $alpha_split"
    else
        "$PYTHON_BIN" scripts/prepare_split.py \
            --data_root "$DATA_ROOT" \
            --output_dir "$SPLIT_DIR" \
            --source_split_policy official_aod4_v6 \
            --partition dirichlet \
            --alpha 0.1 \
            --num_clients 3 \
            --seed "$seed"
    fi

    iid_split="${SPLIT_DIR}/split_official_v6_iid_c3_s${seed}.json"
    iid_yolo="${SPLIT_DIR}/yolo_official_v6_iid_c3_s${seed}"
    if [[ -e "$iid_split" || -e "$iid_yolo" ]]; then
        if [[ ! -f "$iid_split" || ! -d "$iid_yolo" ]]; then
            echo "[ERROR] Partial IID split artifact pair: $iid_split / $iid_yolo" >&2
            exit 1
        fi
        echo "[SKIP] Existing IID split pair: $iid_split"
    else
        "$PYTHON_BIN" scripts/prepare_split.py \
            --data_root "$DATA_ROOT" \
            --output_dir "$SPLIT_DIR" \
            --source_split_policy official_aod4_v6 \
            --partition iid \
            --num_clients 3 \
            --seed "$seed"
    fi
done

for seed in 42 43 44; do
    test -f "${SPLIT_DIR}/split_official_v6_dirichlet_a0.1_c3_s${seed}.json"
    test -d "${SPLIT_DIR}/yolo_official_v6_dirichlet_a0.1_c3_s${seed}"
    test -f "${SPLIT_DIR}/split_official_v6_iid_c3_s${seed}.json"
    test -d "${SPLIT_DIR}/yolo_official_v6_iid_c3_s${seed}"
done
echo "[PASS] Six required alpha=0.1/IID manifest+YOLO artifact pairs are present"
