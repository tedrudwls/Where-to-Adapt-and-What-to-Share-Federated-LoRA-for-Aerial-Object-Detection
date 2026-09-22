# Held-out test qualitative comparison

This is a **two-stage, read-only analysis of historical primary inputs**, not a new training run or an AP re-evaluation. Run `select` before loading any model. It fixes two official AOD-4 v6 test images using only ground-truth annotations and the audited source-group identity. `render` then applies five validation-selected FL checkpoints to those exact images. Neither stage changes the primary result JSONs, checkpoints, split manifests, or dataset. Both refuse an existing output directory.

## Selection and comparison protocol

- Partition: Dirichlet alpha 0.4, training/partition seed 42, three clients.
- Eligible image: official test image whose Roboflow-filename/SHA-256 connected source component does **not** occur in train or validation. This makes the *two illustrated examples* source-disjoint; it does not alter the official-test AP tables or make the entire official split source-disjoint.
- Two predetermined strata: an image with a drone annotation and an image with a helicopter annotation. Within each class, the selected image has a largest target box whose normalized area is closest to the class median on a log-area scale; image ID breaks ties. The two image IDs must differ. This ground-truth-only rule avoids selecting cases based on model successes or failures. Predictions are not available at this stage.
- Row 1 (placement): GT, FedLoRA-AB Decoder-only, FedLoRA-AB Backbone-only, FedLoRA-AB Backbone+Decoder.
- Row 2 (sharing): GT, FedLoRA-AB, FedLoRA-A (shared A/local B), FedLoRA-B (shared B/local A). Each row compares its methods on **one identical image**; the two rows use different test images.
- All five methods use rank 8, the same split, selected-best checkpoint, 640-pixel input and display confidence threshold 0.25. The threshold is **for rendering**, not the COCO AP computation. Personalized endpoints restore the factor belonging to the selected image's client.

## Server commands

From a checkout of this release-draft branch with the historical project and dataset still at their verified paths:

```bash
cd /home/gpuadmin/kim/fedsalora
source .venv/bin/activate

CODE_DIR=/path/to/Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection
PROJECT=/home/gpuadmin/kim/fedsalora
DATA_ROOT=/home/gpuadmin/kim/project2/data/aod4/AOD4/Images
SPLIT_FILE="$PROJECT/data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json"
TAG=$(date -u +%Y%m%dT%H%M%SZ)
SELECTION_DIR="$PROJECT/review_evidence/qualitative_selection_$TAG"
RENDER_DIR="$PROJECT/review_evidence/qualitative_render_$TAG"

python3 "$CODE_DIR/scripts/render_test_qualitative.py" select \
  --data-root "$DATA_ROOT" \
  --split-file "$SPLIT_FILE" \
  --output-dir "$SELECTION_DIR"

# Inspect and freeze selection.json before any inference.
python3 -m json.tool "$SELECTION_DIR/selection.json" | less
sha256sum "$SELECTION_DIR/selection.json"

CUDA_VISIBLE_DEVICES=2 python3 "$CODE_DIR/scripts/render_test_qualitative.py" render \
  --selection "$SELECTION_DIR/selection.json" \
  --split-file "$SPLIT_FILE" \
  --data-root "$DATA_ROOT" \
  --project-dir "$PROJECT" \
  --results-root "$PROJECT/results/official_v6" \
  --checkpoint-index "$CODE_DIR/artifacts/checkpoint_index.json" \
  --model-weights "$PROJECT/rtdetr-l.pt" \
  --device cuda:0 \
  --output-dir "$RENDER_DIR"
```

`CUDA_VISIBLE_DEVICES=2` makes physical GPU 2 appear as `cuda:0` to this process. Do not reuse an existing output directory; a rerun must get a new tag. The artifacts record the selected image IDs and source groups, ground truth, selected checkpoint hashes/rounds, rendering parameters, and displayed prediction boxes. Preserve the executed command or terminal log separately. The `best_federated.pt` paths must exist and match the 96-checkpoint SHA-256 index; there is no fallback to `last_federated.pt`.

Do not use `main.py --resume` or `evaluate_federated_checkpoint()` for this figure: those paths can rewrite primary result files. PyTorch checkpoints are pickle-based; load only the trusted, hash-verified historical files.

## Reporting boundary

The composite is an **illustration** of detector behavior, not a substitute for the three-seed AP and communication tables. Do not replace an image after seeing the predictions merely because another looks better. If the selected case is visually ambiguous, report that limitation or specify a new, outcome-blind selection protocol as a separately labeled exploratory figure. Publish the selection and prediction metadata alongside the figure. The final paper caption must disclose that these are two seed-42, source-disjoint *examples* from the official test set, plus the fixed display threshold and best-checkpoint rule.
