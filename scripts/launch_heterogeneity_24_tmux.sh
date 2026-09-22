#!/usr/bin/env bash
# Launch exactly 24 frozen heterogeneity runs as six four-job tmux queues.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/splits}"
QUEUE_SCRIPT="${PROJECT_DIR}/scripts/run_heterogeneity_queue.sh"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python3}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${PROJECT_DIR}/rtdetr-l.pt}"
export PROJECT_DIR SPLIT_DIR PYTHON_BIN MODEL_WEIGHTS

command -v tmux >/dev/null || {
    echo "[ERROR] tmux is not installed or not on PATH" >&2
    exit 1
}
[[ -f "$QUEUE_SCRIPT" ]] || {
    echo "[ERROR] Missing queue wrapper: $QUEUE_SCRIPT" >&2
    exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
    echo "[ERROR] Missing virtual-environment Python: $PYTHON_BIN" >&2
    exit 1
}
[[ -f "$MODEL_WEIGHTS" ]] || {
    echo "[ERROR] Missing pretrained checkpoint: $MODEL_WEIGHTS" >&2
    exit 1
}

sessions=(
    het_a01_s42 het_iid_s42
    het_a01_s43 het_iid_s43
    het_a01_s44 het_iid_s44
)
for session in "${sessions[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[ERROR] Existing tmux session: $session" >&2
        echo "[STOP] Inspect that session; no new queue was launched." >&2
        exit 1
    fi
done

for seed in 42 43 44; do
    alpha_split="${SPLIT_DIR}/split_official_v6_dirichlet_a0.1_c3_s${seed}.json"
    alpha_yolo="${SPLIT_DIR}/yolo_official_v6_dirichlet_a0.1_c3_s${seed}"
    iid_split="${SPLIT_DIR}/split_official_v6_iid_c3_s${seed}.json"
    iid_yolo="${SPLIT_DIR}/yolo_official_v6_iid_c3_s${seed}"
    if [[ ! -f "$alpha_split" || ! -d "$alpha_yolo" || \
          ! -f "$iid_split" || ! -d "$iid_yolo" ]]; then
        echo "[ERROR] Missing or wrong-type split/YOLO artifact for seed=$seed" >&2
        echo "[STOP] Run scripts/prepare_heterogeneity_splits.sh first." >&2
        exit 1
    fi
done

launch_queue() {
    local session="$1"
    local gpu="$2"
    local condition="$3"
    local seed="$4"
    tmux new-session -d -s "$session" \
        "cd '$PROJECT_DIR' && export CUDA_VISIBLE_DEVICES='$gpu' && \
         exec bash scripts/run_heterogeneity_queue.sh '$condition' '$seed'"
    tmux set-window-option -t "${session}:0" remain-on-exit on
    echo "[LAUNCHED] session=$session gpu=$gpu condition=$condition seed=$seed"
}

# Two queues per physical GPU; each queue executes its four methods serially.
launch_queue het_a01_s42 0 alpha0.1 42
launch_queue het_iid_s42 0 iid 42
launch_queue het_a01_s43 1 alpha0.1 43
launch_queue het_iid_s43 1 iid 43
launch_queue het_a01_s44 2 alpha0.1 44
launch_queue het_iid_s44 2 iid 44

echo "[PASS] Six tmux queues launched (4 sequential runs each = 24 total)"
tmux list-panes -a \
    -F 'session=#{session_name} dead=#{pane_dead} exit=#{pane_dead_status}' \
    | grep '^session=het_' || true
