# Validation-selected checkpoints

The [checkpoint index](../artifacts/checkpoint_index.json) describes **96** selected models (4,554,894,872 bytes total), each with an experiment ID, historical relative path, byte size, SHA-256 digest, selected round/epoch, prerequisite pretrained-model digest and split-manifest digest. It is a checksum inventory, **not** a model download. No `.pt` binary is in this source tree.

## Planned artifact layout

After the server files have passed the read-only verifier and redistribution terms are fixed, each selected checkpoint can be uploaded as a uniquely named GitHub Release asset; the intended name is the index field `release_asset_name`. The raw 96-file result JSON bundle should be a separate versioned asset. Publish an SHA-256 checksum file alongside each asset set. Do not add checkpoint binaries to ordinary Git history. `rtdetr-l.pt` must be verified and handled under its own redistribution terms; if it is not released, documentation must give an authoritative acquisition route and its expected SHA-256.

The recorded pretrained model SHA-256 is:

```text
6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f  rtdetr-l.pt
```

On the artifact host, run a **read-only** inventory check from the project root before any packaging or upload:

```bash
PROJECT_DIR=/path/to/this/repository
python3 scripts/verify_checkpoint_assets.py \
  --project-dir "$PROJECT_DIR"
sha256sum "$PROJECT_DIR/rtdetr-l.pt"
```

The user ran this verifier on the artifact host using draft PR code and reported `[PASS] checkpoints 96/96 verified; bytes=4554894872`; the pretrained SHA-256 also matched the recorded value. The verifier does not rewrite checkpoints, result JSONs, or split manifests. Repeat this check immediately before asset upload, then compare every public download to the index. Never load an untrusted PyTorch pickle checkpoint: `torch.load` can execute code.

## Model-state semantics

`best_full.pt` files are full selected solo/centralized detector states. `best_federated.pt` files are federated selected states; factor-sharing methods require the recorded client-specific factor states for personalized evaluation. The pretrained base checkpoint and the exact code/dependency version are required to rebuild LoRA models correctly. A selected checkpoint contains state needed for **inference/evaluation**, not exact resumption of training: optimizer moments, scheduler phase and all RNG states are not supplied as a complete restart snapshot. Selected rounds/epochs are chosen by validation performance; test data did not choose them.

The public [qualitative record](QUALITATIVE_TEST.md) consists of two static seed-43/client-1 figures and path-sanitized metadata for three validation-selected rank-8 endpoints. It is not a general checkpoint evaluation or AP recomputation interface. Do not use `main.py --resume` against archived primary directories as an evaluation shortcut: that path writes evaluation manifests, logs, plots and result JSONs. A separate [read-only evaluation vertical slice](READ_ONLY_EVALUATION.md) implements temporary test-only reconstruction for the indexed seed-42 FedLoRA-A checkpoint without editing the path-bound historical manifest. The authors report that its pinned GPU/model path passed the documented artifact-host acceptance command on 2026-09-23.

The next release stage is the separate
[paper-core checkpoint set](PAPER_CORE_RELEASE.md): Full FT, FedLoRA-AB,
FedLoRA-A and FedLoRA-B for seeds 42/43/44 under the primary alpha-0.4
protocol. It is an exact 12-checkpoint subset of this index and does not replace
the representative replay artifact or imply that all 96 checkpoints have been
published. A read-only 12/12 host audit has now pinned each path-sanitized
public byte size, SHA-256 value and tensor fingerprint in the release
specification. This identity pinning is a prepublication gate, not a download
claim; the archive remains unpublished until the clean build, GPU replay and
public re-download checks pass.

The 96 checkpoint files are on the private artifact host and passed the user-reported host-side checksum check, but have **not yet been published as download assets or independently rehashed after download**. Links will be added only after publication and public-download verification. Until then, do not interpret an index record as proof that the corresponding binary is publicly available.

## Minimal representative release gate

Before uploading all 96 binaries, the project uses one small end-to-end release
gate for `seed_42/fl_fedsa_lora_r8_a0.4`. The source checkpoint embeds two
historical pretrained-weight paths, so it is not uploaded byte-for-byte. The
[representative release procedure](REPRESENTATIVE_RELEASE.md) verifies the
historical SHA-256, changes only those two path strings, proves identical tensor
state, packages a compact path-free test manifest, and requires a clean public
re-download plus GPU AP replay. The historical and sanitized public checkpoint
identities are recorded separately; the checkpoint index remains the inventory
of immutable historical files.
