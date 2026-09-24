# Reproduce the experiments

This is the **frozen reported protocol**, not an invitation to tune on the test set. The original runs were executed on NVIDIA RTX A6000 GPUs with Python 3.9.18, PyTorch 2.5.1+cu124, torchvision 0.20.1 and Ultralytics 8.4.126. Reproducing the historical binary environment may require an older compatible Python/CUDA installation. The pinned `requirements.txt` records the project-level packages, but does not include the NVIDIA driver or OS package versions. An installed package with the same version is not by itself proof of numerical bit identity.

## 1. Environment and prerequisites

Use a clone or checkout of the released commit. Supply the original AOD-4 COCO images and a verified RT-DETR-L pretrained checkpoint separately. Do not move protected historical result directories into a new run directory.

```bash
cd /absolute/path/to/this/repository
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
# Install the PyTorch/torchvision build appropriate to your CUDA driver first.
python3 -m pip install -r requirements.txt

export PROJECT_DIR="$PWD"
export DATA_ROOT=/absolute/path/to/AOD4/Images
export SPLIT_DIR="$PROJECT_DIR/data/splits"
export RESULTS_ROOT="$PROJECT_DIR/results/official_v6"
export LOG_ROOT="$PROJECT_DIR/logs/official_v6"
export MODEL_WEIGHTS=/absolute/path/to/verified/rtdetr-l.pt
export PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"

sha256sum "$MODEL_WEIGHTS"
python3 -m unittest discover -s tests
```

The expected pretrained digest is listed in [Checkpoints](CHECKPOINTS.md). Do not use an unverified `.pt` file. `source` and shell environment variables in the examples are for Bash; adapt them to your shell. For a CPU-only basic code test, no GPU is necessary, but training and the preflight forward/backward smoke test require a compatible CUDA installation.

## 2. Generate immutable splits

The original data root must contain `train`, `val`, and `test` COCO directories as described in [Dataset](DATASET.md). Generate all nine published partition manifests locally; they are path-bound and should not be copied from the old server as runnable inputs.

```bash
for seed in 42 43 44; do
  python3 scripts/prepare_split.py \
    --data_root "$DATA_ROOT" --output_dir "$SPLIT_DIR" \
    --source_split_policy official_aod4_v6 \
    --partition dirichlet --alpha 0.4 --num_clients 3 --seed "$seed"
  python3 scripts/prepare_split.py \
    --data_root "$DATA_ROOT" --output_dir "$SPLIT_DIR" \
    --source_split_policy official_aod4_v6 \
    --partition dirichlet --alpha 0.1 --num_clients 3 --seed "$seed"
  python3 scripts/prepare_split.py \
    --data_root "$DATA_ROOT" --output_dir "$SPLIT_DIR" \
    --source_split_policy official_aod4_v6 \
    --partition iid --num_clients 3 --seed "$seed"
done
```

Preparation verifies the annotations, image decodability and inventory, writes a source-group-aware client assignment and creates YOLO-format intermediate labels/symlinks for the RT-DETR training interface. Do **not** edit generated YAML or manifest digests by hand. Reusing the same `DATA_ROOT`, source policy and seed should reproduce the same assignment; the released historical manifests are a provenance comparison, not a replacement for local generation.

Before a long run, exercise the exact model and one labeled loss batch:

```bash
python3 scripts/preflight.py \
  --data_root "$DATA_ROOT" \
  --split_file "$SPLIT_DIR/split_official_v6_dirichlet_a0.4_c3_s42.json" \
  --model_weights "$MODEL_WEIGHTS" \
  --partition dirichlet --dirichlet_alpha 0.4 \
  --num_clients 3 --partition_seed 42 --seed 42 \
  --mode fl --fl_method lora --lora_rank 8 --lora_alpha 16 \
  --device cuda
```

Run the same preflight for `full_ft`, `fedsa_lora` and `fixed_share_b_lora` if those methods will be trained. Preflight is a diagnostic, not a substitute for an end-to-end training run.

## 3. Primary 36 experiments

The release defines three paired replicates, one per `seed`. For each replicate: six local baselines (three clients × full FT/LoRA), two centralized references, and four FL policies. Use independent GPUs or schedule serially; do not have two processes write to the same experiment directory.

```bash
for seed in 42 43 44; do
  export TRAIN_SEED="$seed" PARTITION_SEED="$seed"
  bash scripts/run_solo.sh all full_ft
  bash scripts/run_solo.sh all lora
  bash scripts/run_centralized.sh all
  for method in full_ft lora fedsa_lora fixed_share_b_lora; do
    bash scripts/run_fl.sh "$method" 0.4 8
  done
done
```

The FL protocol is three full-participation clients, **20 rounds × five local epochs**, with the optimizer state reset each round but a continuous global-step warmup/cosine schedule. Solo and centralized use 100 epochs. Training uses batch size eight, 640-pixel images, AdamW, no AMP, no early stopping (`--patience 0`), and validation-based best-checkpoint selection. `run_*.sh` scripts skip only when the existing result matches the frozen CLI protocol. They fail on conflicting/incomplete results rather than overwrite them unless `FORCE_RERUN=1` is explicitly set; preserving primary data is recommended.

## 4. Additional experimental grids

Placement-only (18 **new** runs) compares `decoder_only` and `backbone_only` for three sharing policies and three seeds; `both` reuses nine primary rank-8 FL checkpoints. Rank sensitivity adds rank 4 and 16 across the three sharing policies and three seeds (18 new runs); rank 8 reuses primary checkpoints. Use the single-cell `run_lora_grid.sh` runner for all three policies. The historical `run_target_ablation.sh` and `run_rank_sensitivity.sh` wrappers cover **FedLoRA-A only**; do not mistake them for the complete three-policy grid. The 24 heterogeneity runs compare four FL methods across three seeds for each of IID and Dirichlet α=0.1.

```bash
for seed in 42 43 44; do
  for method in lora fedsa_lora fixed_share_b_lora; do
    for target in decoder_only backbone_only; do
      bash scripts/run_lora_grid.sh "$method" 8 "$target" "$seed"
    done
    for rank in 4 16; do
      bash scripts/run_lora_grid.sh "$method" "$rank" both "$seed"
    done
  done
done
```

Set `DRY_RUN=1` to inspect each resolved experiment ID and command without writing anything. The command refuses to overwrite nonempty or conflicting experiment outputs. Its names were checked against all 96 recorded selected-checkpoint experiment IDs, but the code has not been re-executed on the historical server in this local draft.

For a single heterogeneity cell:

```bash
bash scripts/run_heterogeneity_cell.sh alpha0.1 lora 42
bash scripts/run_heterogeneity_cell.sh iid lora 42
```

For the six required IID/α=0.1 split pairs and their read-only result gate:

```bash
bash scripts/prepare_heterogeneity_splits.sh
# Train desired cells or use the tmux queue only after checking GPU capacity.
bash scripts/verify_heterogeneity_24.sh
```

`launch_heterogeneity_24_tmux.sh` starts six queues (two per physical GPU, four cells per queue) and assumes three suitable devices and enough memory for concurrency. It is **not** a portable GPU scheduler. Set `CUDA_VISIBLE_DEVICES`/device mapping and inspect its queue order before using it on a different server. Other optional alpha values (`0.5`, `1.0`) are supported by scripts but are not part of the reported 96 selected models.

## 5. Evaluation, aggregation and audits

Each training run writes a result JSON and validation-selected checkpoint under `RESULTS_ROOT/seed_<seed>/<experiment>/`. Plots and logs are diagnostics; the JSON and selected checkpoint are the primary record. `scripts/result_complete.py` checks status/protocol/split identity. Aggregation reads result JSONs; it does not retrain models. See [Results](RESULTS.md) for the distinction among own-client, common-test and class AP.

The source release includes `scripts/aggregate_results.py`, `scripts/summarize_target_ablation.py`, `scripts/summarize_unseen_class_case.py`, and the MIA audit scripts. Run analyses on **copies or separate output directories** after the primary result gate has passed. The frozen MIA analyses are endpoint diagnostics, not formal privacy certification; see [Security audit scope](SECURITY_AUDIT.md).

The 96 original selected checkpoints and complete result JSON bundle are not presently in this source tree. A verified [paper-core Release](https://github.com/tedrudwls/Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection/releases/tag/paper-core-checkpoints-v1.0.0) provides the 12 primary checkpoints; the remaining 84 checkpoints and the complete result JSON bundle are not yet public. A deliberately narrow [read-only evaluation vertical slice](READ_ONLY_EVALUATION.md) is included for the indexed seed-42 FedLoRA-A checkpoint. It verifies relocated test data, reconstructs three personalized endpoints in a temporary test-only tree, recomputes own-client/common-test AP, and compares against a compact frozen reference. The pre-release replay passed on 2026-09-23, and on 2026-09-24 the same checkpoint was anonymously re-downloaded inside the paper-core archive and again matched the frozen metrics with maximum absolute error `0.0`; see the [public release acceptance receipt](../artifacts/public_release_acceptance_receipt.json). `main.py --resume` writes into experiment directories and must not be used as a harmless checkpoint viewer. A full recomputation from scratch remains possible if the dataset, verified pretrained weight, hardware and software environment are available. Exact bitwise replay is not promised across different CUDA installations or GPU scheduling.
