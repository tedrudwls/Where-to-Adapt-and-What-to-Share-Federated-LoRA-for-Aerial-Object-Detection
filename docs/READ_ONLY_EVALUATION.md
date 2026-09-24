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
- In author-side mode it verifies the immutable schema-v7 split and optional
  archived result. In public mode it verifies the code-pinned, path-free replay
  manifest. Both modes verify the checkpoint and pretrained-model identities,
  test annotation SHA-256, and all 2,241 test-image SHA-256 values before model
  construction.
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
| historical `best_federated.pt` (author-side mode) | `3ed025419506009465add698da75fef941c20c0b16eb347bb224820781b9cac4`, 3,316,516 bytes |
| sanitized public `best_federated.pt` (public replay mode) | `391205473ad8de24af56ba1b566e54a6f305dd0b79806cb468583e84e464fa14`, 3,316,452 bytes; tensor fingerprint `b42e1e811238caf1ec76e788547dbcf46d8f390b3b0e63864cf3d303e88c6d4f` over 231 tensors |
| `rtdetr-l.pt` | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| `split_official_v6_dirichlet_a0.4_c3_s42.json` | `74a45a37f4b05c474564993ff15f5548875f3e767abc19a0d2d30f195e1754c5` |
| archived `fl_results.json` | `ed2b53790bc3a44a157ce61f078332723be388744af54ec315cac3badbfa2c4d`, 5,297,013 bytes |
| public replay manifest | `ffe7b2932dfcfb0027a8e607abb3d8d5c8a5241d549b280524b77be5f3b44c88`, 472,024 bytes |
| committed evaluation reference | `4f6d33eeb255a96d0f49c51600dcf546a2ac5d96e782bd9826094f848e98dbef` |

The checkpoint index, compact expected-metric record, and path-free replay
manifest are committed under `artifacts/`. The checkpoint, complete split
manifest, archived result and pretrained weight are not embedded in ordinary
Git history.

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
gates. On 2026-09-23, the user ran the command above in the pinned artifact-host
environment: all 2,241 test images were verified, the final integrity gate
passed, and the recomputed client-local/common-test metrics matched the archived
reference within the frozen absolute tolerance (`1e-6`).
This validates the vertical slice for the exact frozen artifacts and pinned
environment. The same sanitized checkpoint is now public inside the verified
[paper-core Release](https://github.com/tedrudwls/Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection/releases/tag/paper-core-checkpoints-v1.0.0).

The sanitized public-checkpoint form was then tested from evaluator source
commit `461bb35b3d22f3d44da9c68e4cb9ead5ebad4761` on the same recorded RTX A6000
artifact-host environment. Thirty-five targeted tests passed, all 2,241 test
images and the public checkpoint/replay/pretrained identities were verified,
and the final integrity gate passed. The recomputed client-local macro AP was
`0.6190019159385891`, the common pooled-test macro AP was
`0.5620845765652397`, and the maximum absolute reference error was `0.0`.
The path-free [pre-release acceptance receipt](../artifacts/representative_public_gpu_acceptance.json)
records the complete AP/AP50/AP75 values, hashes, runtime and reporting scope.
This is an author-run evaluation replay for one checkpoint and one environment;
it is not training reproduction or independent public-download verification.

## Public replay mode

`--replay-manifest` is mutually exclusive with the author-side `--split-file`.
It removes the need to publish the historical path-bearing schema-v7 manifest.
`--reference-result` is also optional in this mode: recomputed metrics are
compared with the hash-pinned compact evaluation reference, while the report
explicitly records `archived_result_verified: false`.

The Phase-1 build fixed the sanitized checkpoint identity listed above. That
checkpoint first passed the author-run pre-release GPU replay recorded in the
[pre-release acceptance receipt](../artifacts/representative_public_gpu_acceptance.json).
It was then included in the paper-core Release, anonymously re-downloaded, and
replayed again with maximum absolute error `0.0`, as recorded in the
[public release acceptance receipt](../artifacts/public_release_acceptance_receipt.json).
The command below is the frozen replay procedure for an extracted public
paper-core archive:

```bash
: "${PROJECT_DIR:?Set PROJECT_DIR to a clean checkout of the release source commit}"
: "${BUNDLE_ROOT:?Set BUNDLE_ROOT to the directory where the paper-core archive was extracted}"
: "${DATA_ROOT:?Set DATA_ROOT to the verified AOD-4 Images directory}"
: "${MODEL_WEIGHTS:?Set MODEL_WEIGHTS to the verified rtdetr-l.pt file}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT=$(mktemp)

cd "$PROJECT_DIR"
"$PYTHON_BIN" \
  scripts/evaluate_checkpoint.py \
  --checkpoint \
    "$BUNDLE_ROOT/paper-core-checkpoints/checkpoints/seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt" \
  --model-weights "$MODEL_WEIGHTS" \
  --replay-manifest \
    "$PROJECT_DIR/artifacts/seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  > "$REPORT"

"$PYTHON_BIN" - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["status"] == "pass"
assert report["integrity_gate"]["status"] == "pass"
assert report["reference_comparison"]["passed"] is True
assert report["inputs"]["checkpoint"]["variant"] == "public_sanitized"
assert report["inputs"]["test_images_verified"] == 2241
print("[PASS] Sanitized public checkpoint reproduced the frozen metrics")
PY
```

The public gate completed on 2026-09-24. The published archive was downloaded
without authentication into a new directory, its outer checksum and all 12
checkpoint identities passed structural verification, and the extracted
representative checkpoint reproduced the frozen metrics over all 2,241 test
images with maximum absolute error `0.0`. This verifies evaluation replay for
one included checkpoint; it does not claim AP replay for the other 11
checkpoints or reproduce training. See the
[paper-core release record](PAPER_CORE_RELEASE.md) and
[final acceptance receipt](../artifacts/public_release_acceptance_receipt.json).
