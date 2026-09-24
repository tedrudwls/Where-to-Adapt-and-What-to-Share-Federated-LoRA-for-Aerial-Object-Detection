# Paper-core checkpoint release

This release is the smallest checkpoint set that represents the paper's main
federated comparison. It is separate from the existing seed-42 FedLoRA-A
representative replay release, which remains an end-to-end artifact-pipeline
smoke test.

## Scope

The paper-core release contains the validation-selected checkpoints for four
methods and three paired training/partition seeds under the official
Dirichlet-alpha-0.4 protocol:

| Public method | Internal method | Seeds | LoRA setting |
| --- | --- | --- | --- |
| FL Full FT | `full_ft` | 42, 43, 44 | not applicable |
| FedLoRA-AB | `lora` | 42, 43, 44 | rank 8, Backbone+Decoder |
| FedLoRA-A | `fedsa_lora` | 42, 43, 44 | rank 8, Backbone+Decoder |
| FedLoRA-B | `fixed_share_b_lora` | 42, 43, 44 | rank 8, Backbone+Decoder |

The exact allowlist is therefore 12 checkpoints. Their immutable historical
identities are selected from
[`artifacts/checkpoint_index.json`](../artifacts/checkpoint_index.json); their
combined historical size is 423,860,180 bytes (404.225 MiB).

The proposed public assets are:

```text
fedlora-paper-core-checkpoints-v1.0.0.tar.gz
fedlora-paper-core-checkpoints-v1.0.0.tar.gz.sha256
```

under the immutable release tag `paper-core-checkpoints-v1.0.0`. Checkpoint
binaries are release assets and are not committed to ordinary Git history.

## Two-phase publication gate

Historical PyTorch checkpoints contain two reviewed absolute paths to the
external pretrained weight. Public files must therefore be derived rather than
uploaded byte-for-byte.

1. **Read-only discovery.** Verify all 12 historical sizes and SHA-256 values,
   load each checkpoint with restricted loading, validate its method/state
   contract, replace only the two reviewed pretrained-weight paths in a copy,
   prove that every tensor is byte-identical, and emit a path-free candidate
   identity report. The source checkpoints are hashed before and after the
   operation.
2. **Identity pinning and release build.** Review and commit the 12 public file
   identities from phase 1. A clean build must reproduce each exact public
   byte size, SHA-256 digest, tensor count and tensor fingerprint before it may
   create the archive.

This separation prevents a self-consistent but unintended checkpoint rewrite
from being accepted merely because the archive manifest and checksums were
rewritten at the same time.

Phase 1 is complete. The committed
[path-free audit report](../artifacts/paper_core_input_audit.json) covered all 12 checkpoints,
kept all 14 protected inputs byte-identical, and recorded the following
portable identity:

```text
report bytes:   37,737
report SHA-256: c41bd0091da3ee727477df5c96377147f04316340794fef5d0c681cbccf80be5
runtime:        Python 3.9.18 / PyTorch 2.5.1+cu124
historical:     423,860,180 bytes
public files:   423,859,412 bytes
```

The 12 resulting public sizes, SHA-256 values, tensor counts and tensor
fingerprints are now pinned in
[`artifacts/paper_core_checkpoint_release_spec.json`](../artifacts/paper_core_checkpoint_release_spec.json).
The original host-generated checksum sidecar contained a server absolute path,
so that file was not retained. A canonical basename-only
[`paper_core_input_audit.json.sha256`](../artifacts/paper_core_input_audit.json.sha256)
sidecar was generated from the independently verified digest and committed.

## Clean bundle build

Run the builder from a clean checkout at an immutable commit while pointing
`--source-project` to the historical artifact tree. It restricted-loads and
processes one checkpoint set at a time, reproduces every pinned public identity,
streams the deterministic tar.gz, re-hashes all protected inputs, and performs
the structural verifier before an atomic output-directory rename.

```bash
python3 scripts/build_paper_core_release_bundle.py \
  --source-project /path/to/historical/fedsalora \
  --output-dir /path/to/new/paper-core-release-output

python3 scripts/verify_paper_core_release_bundle.py \
  --archive /path/to/new/paper-core-release-output/fedlora-paper-core-checkpoints-v1.0.0.tar.gz \
  --checksum /path/to/new/paper-core-release-output/fedlora-paper-core-checkpoints-v1.0.0.tar.gz.sha256
```

The verifier checks the outer digest on the same open file descriptor before
opening its gzip/tar stream and hashes tar members incrementally. It never
deserializes checkpoint data. The separate GPU replay gate remains required
before publication.

## Intended archive boundary

```text
paper-core-checkpoints/
├── BUNDLE_MANIFEST.json
├── SHA256SUMS
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── checkpoints/
│   └── <12 path-sanitized checkpoint files>
└── metadata/
    ├── paper_core_checkpoint_release_spec.json
    └── paper_core_input_audit.json
```

The archive does **not** include AOD-4 images or annotations, the upstream
`rtdetr-l.pt` weight, historical result JSONs, path-bearing split manifests,
optimizer state sufficient for exact training resumption, or the remaining 84
ablation checkpoints.

## Interpretation boundary

- These are inference/evaluation checkpoints selected by validation
  performance. They are not exact training-resumption snapshots.
- The set supports the paper's primary method-by-seed comparison. Placement,
  rank, IID and alpha-0.1 ablations remain indexed separately.
- The previously published seed-42 FedLoRA-A release is preserved unchanged;
  it is not overwritten or relabeled as FedLoRA-AB.
- Structural and tensor-identity verification does not by itself reproduce AP.
  The existing seed-42 FedLoRA-A public replay is the current GPU acceptance
  test. Additional checkpoint replay claims must be made only after they are
  actually run and recorded.
- AOD-4 and the upstream pretrained model remain external inputs governed by
  their own terms and expected hashes.

## Release acceptance

Before publication, the maintainers must complete all of the following:

1. **Completed:** obtain a 12/12 passing read-only discovery report;
2. **Completed:** commit the reviewed, path-free public identity specification;
3. build from a clean immutable source commit using streaming archive I/O;
4. verify the outer checksum and every exact archive member identity;
5. extract into a fresh directory and run at least the already documented
   seed-42 FedLoRA-A GPU replay from the bundled checkpoint;
6. upload the two assets to a draft GitHub Release;
7. download both assets through the public URL into a clean directory and
   repeat the checksum, structural and GPU gates; and
8. publish a path-free release receipt without editing the tagged commit.
