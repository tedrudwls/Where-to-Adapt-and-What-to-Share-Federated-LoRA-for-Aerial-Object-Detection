# Representative checkpoint release procedure

This procedure publishes one minimal, independently testable artifact before
the full 96-checkpoint release. The target is the seed-42 FedLoRA-A rank-8
Backbone+Decoder checkpoint used by the read-only evaluation vertical slice.
FedLoRA-A is selected here because its checkpoint exercises both shared state
and three client-personalized local-factor states; it is not the paper's main
accuracy configuration.

> **Status (2026-09-24):** no standalone representative archive was published.
> Instead, this exact sanitized checkpoint is included in the verified
> [paper-core-checkpoints-v1.0.0 Release](https://github.com/tedrudwls/Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection/releases/tag/paper-core-checkpoints-v1.0.0)
> and passed an unauthenticated re-download, structural verification, and GPU
> replay. See the [final acceptance receipt](../artifacts/public_release_acceptance_receipt.json).
> The remainder of this document preserves the standalone preparatory procedure
> for provenance; it is not a claim that those standalone assets exist.

## Public asset boundary

The planned GitHub Release contains two files:

```text
fedlora-representative-replay-seed42-v1.0.0.tar.gz
fedlora-representative-replay-seed42-v1.0.0.tar.gz.sha256
```

The archive contains:

```text
representative-replay/
├── BUNDLE_MANIFEST.json
├── SHA256SUMS
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── checkpoint/
│   └── seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt
└── metadata/
    └── seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json
```

The raw AOD-4 images, Ultralytics `rtdetr-l.pt`, full historical schema-v7
manifest, and path-bearing 5.3 MB primary result JSON are not included. The
compact replay manifest retains the test annotation identity, all 2,241 test
image hashes, client test assignments, split counts and the historical split
identity without publishing server paths or unnecessary train/validation IDs.

The historical checkpoint itself contains two absolute paths to the pretrained
weight. The builder first verifies the original checkpoint's frozen SHA-256,
then changes only `experiment.model_weights` and
`architecture.model_weight_path` to `external/rtdetr-l.pt`. It reloads the new
file and checks every non-path payload value plus a path/dtype/shape/raw-tensor
fingerprint. The public checkpoint therefore has a new file SHA-256 while its
model tensor state is unchanged.

## Phase 1: build on the artifact host

Use a clean checkout of the preparation branch or, after merge, the immutable
release candidate commit. Run with the historical project's Python environment
so that the exact checkpoint can be loaded with `weights_only=True`.

```bash
set -Eeuo pipefail

: "${SOURCE_PROJECT:?Set SOURCE_PROJECT to the historical project checkout}"
: "${BUILD_PARENT:?Set BUILD_PARENT to a private release-build directory}"
SOURCE_CHECKPOINT="$SOURCE_PROJECT/results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/weights/best_federated.pt"
SOURCE_SPLIT="$SOURCE_PROJECT/data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json"
PYTHON_BIN="$SOURCE_PROJECT/.venv/bin/python3"

BUILD_ROOT=$(mktemp -d "$BUILD_PARENT/representative-release-build.XXXXXX")
git clone --depth 1 \
  --branch codex/public-replay-bundle-p0 \
  https://github.com/tedrudwls/Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection.git \
  "$BUILD_ROOT/repo"

"$PYTHON_BIN" "$BUILD_ROOT/repo/scripts/build_representative_release_bundle.py" bundle \
  --project-dir "$BUILD_ROOT/repo" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --split-file "$SOURCE_SPLIT" \
  --output-dir "$BUILD_ROOT/output/representative-replay"

"$PYTHON_BIN" "$BUILD_ROOT/repo/scripts/verify_representative_release_bundle.py" \
  --archive "$BUILD_ROOT/output/fedlora-representative-replay-seed42-v1.0.0.tar.gz" \
  --checksum "$BUILD_ROOT/output/fedlora-representative-replay-seed42-v1.0.0.tar.gz.sha256" \
  | tee "$BUILD_ROOT/output/verification.json"

echo "build_root=$BUILD_ROOT"
```

The builder is fail-closed: an unexpected original hash, extra absolute path,
changed tensor, dirty tracked checkout, existing output target or modified
protected source input stops the operation. It never writes into the historical
result, split or checkpoint locations.

Phase 1 is complete only after the following identities are reviewed and then
hard-bound in the evaluator/release receipt:

- sanitized public checkpoint SHA-256 and byte size;
- tensor fingerprint and tensor count;
- archive SHA-256 and byte size;
- source Git commit;
- compact replay manifest SHA-256
  `ffe7b2932dfcfb0027a8e607abb3d8d5c8a5241d549b280524b77be5f3b44c88`.

The artifact-host Phase-1 build completed on 2026-09-23 from source commit
`aeb0a31b05387eb43afdfbbd94fa4c3b4fcf5484` with:

| Record | Phase-1 value |
| --- | --- |
| Sanitized checkpoint | 3,316,452 bytes; `391205473ad8de24af56ba1b566e54a6f305dd0b79806cb468583e84e464fa14` |
| Tensor fingerprint | 231 tensors; `b42e1e811238caf1ec76e788547dbcf46d8f390b3b0e63864cf3d303e88c6d4f` |
| Replay manifest | 472,024 bytes; `ffe7b2932dfcfb0027a8e607abb3d8d5c8a5241d549b280524b77be5f3b44c88` |
| Pre-pin archive | 3,143,801 bytes; `c78d42b872e4435a8e7147ddf0c42e4aca3d9f728d8153d51ced2d820bb785f5` |

The checkpoint and tensor identities are now code-pinned. The archive identity
above is evidence for the Phase-1 build only, not a final Release identity: the
required post-pin rebuild embeds a newer source commit and therefore changes
the archive bytes. The final archive digest is recorded only after that rebuild
and the clean public re-download gate.

## External inputs for replay

The release does not redistribute AOD-4 or the upstream pretrained weight.
Obtain AOD-4 from the DOI recorded above and arrange its official COCO export
as documented in [Dataset and splits](DATASET.md). The evaluator verifies the
test annotation plus every one of the 2,241 test-image hashes before model
construction, so a different export fails closed.

With the pinned dependencies installed, Ultralytics can acquire the expected
pretrained filename. Always verify the resulting file before use; a future
upstream artifact with the same name is not accepted automatically.

```bash
mkdir -p external
cd external
python3 - <<'PY'
from ultralytics import RTDETR
RTDETR("rtdetr-l.pt")
PY
printf '%s  %s\n' \
  '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f' \
  'rtdetr-l.pt' | sha256sum -c -
```

## Phase 2: GPU acceptance of the public checkpoint

Do not upload immediately after Phase 1. First pin the newly generated public
checkpoint identity in the evaluator, extract into a fresh directory, and run
the public replay form documented in [Read-only evaluation](READ_ONLY_EVALUATION.md).
It must verify the AOD-4 annotation and all 2,241 test image hashes, reconstruct
the three personalized endpoints, and reproduce the frozen client-local and
common-test metrics within `1e-6`.

After that pin is committed, rebuild the bundle from the clean **post-pin**
commit. The rebuilt public checkpoint SHA-256 and tensor fingerprint must equal
the Phase-1 values; otherwise stop and investigate rather than changing the pin
to fit an unexplained serialization difference. Re-run the structural verifier
and GPU replay on this rebuilt archive. Only this post-pin build is a release
candidate, and its embedded source commit must be the commit whose evaluator
accepts the sanitized checkpoint.

Use the Phase-1 tree recorded by the artifact-host build log only for the first
GPU acceptance run; follow the exact public-mode command in
[Read-only evaluation](READ_ONLY_EVALUATION.md).

The first GPU acceptance run completed on 2026-09-23 using evaluator source
commit `461bb35b3d22f3d44da9c68e4cb9ead5ebad4761` on an NVIDIA RTX A6000. All 35
targeted tests passed; the evaluator verified 2,241 test images, reported a
passing final integrity gate, and reproduced client-local macro AP
`0.6190019159385891` and common pooled-test macro AP
`0.5620845765652397` with maximum absolute error `0.0`. The exact artifact
hashes, full AP/AP50/AP75 values and limitations are stored in the path-free
[pre-release acceptance receipt](../artifacts/representative_public_gpu_acceptance.json).

This completes only the initial sanitized-checkpoint acceptance. The remaining
Phase-2 gate is to rebuild from a clean commit containing that receipt, require
the same public checkpoint SHA-256, size, tensor fingerprint and tensor count,
structurally verify the new archive, and replay the checkpoint extracted from
that new archive. The new archive digest is expected to differ from the
Phase-1 digest because its embedded source commit changes. Do not upload, tag,
or call the artifact public until the rebuilt candidate passes.

## Phase 3: standalone GitHub Release and clean re-download

For the standalone bundle described here, create an immutable release tag only
after Phase 2 passes. Upload the archive
and its outer checksum as GitHub Release assets; do not commit the checkpoint to
ordinary Git history. From a new empty directory:

1. download both assets from the public Release URL;
2. verify the outer `.sha256` file before opening the archive;
3. run `verify_representative_release_bundle.py` on the downloaded files;
4. extract only after the structural verifier passes;
5. verify the separately obtained `rtdetr-l.pt` SHA-256;
6. run the same GPU public replay command on the downloaded checkpoint;
7. commit a path-sanitized release receipt containing the tag, URLs, archive
   SHA-256, public checkpoint SHA-256, source commit and metric comparison.

An uploaded asset is not considered released merely because the GitHub page
exists. The checklist closes only after the clean public re-download passes.
