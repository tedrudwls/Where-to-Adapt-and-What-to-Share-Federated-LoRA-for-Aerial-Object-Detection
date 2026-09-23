# Read-only representative-checkpoint evaluation

This is the P0 evaluation vertical slice for one frozen artifact:

```text
seed_42/fl_fedsa_lora_r8_a0.4
FedLoRA-A (shared A + task head; client-local B)
rank 8, Backbone+Decoder, selected round 20
```

The selective-sharing checkpoint is used for the first release test because it
exercises reconstruction of both shared state and three client-personalized
states. It is a release-integrity smoke test, not a claim that FedLoRA-A is the
paper's best-performing configuration. The paper's reference configuration
remains FedLoRA-AB.

## Safety boundary

`scripts/evaluate_checkpoint.py` is deliberately narrower than the training
entry points.

- It accepts only the experiment above and hard-binds the checkpoint and
  pretrained-model identities in the evaluator code; caller-supplied index or
  reference files cannot authorize a different `.pt` payload.
- It verifies the checkpoint size/SHA-256, pretrained-model SHA-256, immutable
  split-manifest SHA-256, archived-result size/SHA-256, test annotation SHA-256,
  and all 2,241 test-image SHA-256 values before model construction.
- It uses `torch.load(..., map_location="cpu", weights_only=True)` with no
  fallback, as defense in depth only. This is **not** a sandbox or a safe way
  to inspect an untrusted PyTorch file; the pinned historical PyTorch runtime
  has published deserialization advisories. Run this path only with the exact
  author-controlled checkpoint and pretrained-model digests listed below.
- Before either `.pt` is opened by PyTorch/Ultralytics, it is streamed into a
  private temporary file while its code-hard-bound digest is recomputed; model
  construction receives only that verified private copy.
- It permits a relocated AOD-4 root without editing the historical manifest.
- It creates test-only labels, symlinks, YAML and framework caches under a
  temporary directory, then removes that directory.
- It never calls `prepare_data()`, `main.py --resume`, or
  `evaluate_federated_checkpoint()`.
- It evaluates each personalized endpoint on its own client test partition and
  on the identical pooled official test set.
- It captures protected-file and complete test-tree baselines before validation,
  checks them again before model construction, and repeats the checks on every
  exit path.
- It has no persistent output option. The final report is the only stdout JSON;
  Python and file-descriptor-level runtime messages are redirected to stderr.

The CLI does not need the raw train/validation image trees for this AP replay.
Their client sample counts and protocol identity remain bound by the immutable
schema-v7 manifest and checkpoint. This is not a full data-provenance reaudit.

## Required inputs

| Input | Frozen identity |
| --- | --- |
| `best_federated.pt` | `3ed025419506009465add698da75fef941c20c0b16eb347bb224820781b9cac4`, 3,316,516 bytes |
| `rtdetr-l.pt` | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| `split_official_v6_dirichlet_a0.4_c3_s42.json` | `74a45a37f4b05c474564993ff15f5548875f3e767abc19a0d2d30f195e1754c5` |
| archived `fl_results.json` | `ed2b53790bc3a44a157ce61f078332723be388744af54ec315cac3badbfa2c4d`, 5,297,013 bytes |
| committed evaluation reference | `20219c96fc2f3767f8fa9aee96d5d4ee7b7bbd9efc5ce750112f8e512c6b1600` |

The checkpoint index and compact expected-metric record are committed as
`artifacts/checkpoint_index.json` and
`artifacts/evaluation_reference_seed42_fedsa_lora_r8_a04.json`. The checkpoint,
complete split manifest, archived result and pretrained weight are not embedded
in ordinary Git history.

## Artifact-host acceptance command

Run from a clean checkout with the pinned environment. Set the two local paths
explicitly and write stdout outside the archived result tree; the redirection
below is performed by the shell, not by the evaluator.

```bash
PROJECT_DIR=/path/to/this/repository
DATA_ROOT=/path/to/AOD4/Images

cd "$PROJECT_DIR"
source .venv/bin/activate

REPORT=$(mktemp)

python3 scripts/evaluate_checkpoint.py \
  --checkpoint \
    results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/weights/best_federated.pt \
  --reference-result \
    results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/fl_results.json \
  --model-weights rtdetr-l.pt \
  --split-file \
    data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  > "$REPORT"

python3 -m json.tool "$REPORT" >/dev/null
python3 - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["status"] == "pass"
assert report["integrity_gate"]["status"] == "pass"
assert report["reference_comparison"]["passed"] is True
assert report["inputs"]["test_images_verified"] == 2241
print("[PASS] Representative read-only evaluation matched the archived metrics")
print(f"report={sys.argv[1]}")
PY
```

The expected client-local macro AP is `0.6190019159385891`; the expected common
pooled-test macro AP is `0.5620845765652397`. The full client/AP50/AP75 reference
is stored in the compact evaluation record. Metrics use the 0--1 scale.

Exit status `0` means all recomputed metrics matched within the frozen absolute
tolerance (`1e-6`) and the final integrity gate passed. Exit status `1` means
evaluation completed but at least one metric exceeded the tolerance; stdout is
still a machine-readable failure report. Input, dependency, deserialization or
integrity failures return `2` and do not emit a success report.

## Current validation status

The dependency-free unit tests validate relocation, hash ordering, temporary
data construction, JSON-only stdout, metric drift and success/failure read-only
gates. The GPU/model path must still pass the command above on the artifact host
before this CLI is described as a validated public reproduction route.
