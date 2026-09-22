# Historical split provenance

`summary.json` is a compact, mechanically extracted view of the **nine** validated schema-v7 historical split manifests. It preserves the partition regime, paired seed, per-client image/class-instance distribution, background counts, JS divergence, official split counts and COCO annotation digests. `split_manifest_sha256.txt` records the digest of each **full** historical manifest.

The full manifests each contain per-image hash inventory and assignment details, and were intentionally **not** committed as ordinary source files (about 44 MB combined). They remain in the server-side evidence bundle and should be included in a separately versioned research-artifact release after its checksums are verified. This repository does **not** currently provide that complete historical manifest bundle. The compact summary cannot replace the full manifest for runtime validation.

Historical manifests embed the original absolute data path and generated YAML/tree digests. To train on a new checkout, regenerate immutable splits with `scripts/prepare_split.py` as described in [Dataset](../../docs/DATASET.md). Do not rewrite the old manifests or YAML paths.
