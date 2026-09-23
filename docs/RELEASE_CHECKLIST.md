# Publication release gate

This checklist separates **available source/provenance** from **still-private binary artifacts**. A draft pull request does not mean that the paper artifact is complete.

- [x] Curate experiment source, tests, launchers, data manifest provenance and documentation without raw AOD-4 images.
- [x] Record all 96 selected checkpoint names, sizes, SHA-256 digests and prerequisite pretrained/split digests.
- [x] Compare the supplied complete server source archive against the draft: 49 common files are byte-identical and five contain documented portability/comment-only changes. Four recovery/evidence-only server scripts are intentionally excluded pending dependency and disclosure review; see [source comparison](../provenance/SERVER_SOURCE_COMPARISON.md).
- [x] Compile Python, syntax-check shell scripts, run dependency-free tests and validate README/doc links in the local draft.
- [x] License project-authored code, documentation and author-released validation-selected checkpoints under `AGPL-3.0-only`; preserve separate Ultralytics pretrained-model and AOD-4 dataset terms in `THIRD_PARTY_NOTICES.md`.
- [x] On the historical server, the user ran the draft-branch read-only verifier: `[PASS] checkpoints 96/96 verified; bytes=4554894872`. The reported `rtdetr-l.pt` SHA-256 is `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`, matching the recorded digest. This is a user-supplied server result, not an independent download verification; repeat it before the final asset upload.
- [ ] Recheck that the primary result JSONs, frozen split manifests and selected checkpoint hashes did not change during publication preparation.
- [ ] Validate the new seed-42 FedLoRA-A read-only evaluation vertical slice on the original server. Dependency-free relocation/integrity/metric tests pass locally; the GPU/model path must pass the documented command with the two code-hard-bound trusted `.pt` digests before this box is checked. `weights_only=True` is defense in depth, not permission to load an untrusted PyTorch file. Do not use `main.py --resume` to evaluate archived primary results because it writes result files.
- [x] Freeze the seed-43/client-1 p10/p50/p90 cases with the ground-truth-only, source-disjoint rule; render the three validation-selected rank-8 endpoints at a fixed 0.25 display threshold; retain displayed false positives; publish the p10/p50 static figures and path-sanitized selection/prediction metadata. These figures are illustrations and do not replace the full-partition AP result; see [the qualitative record](QUALITATIVE_TEST.md).
- [ ] Publish the 96 uniquely named checkpoint binaries and original result JSON bundle as versioned, checksummed assets, then verify every public download and link the immutable release tag in the README. The locally received 96-JSON archive digest is recorded in `artifacts/result_bundle_sha256.txt`; it is not yet a public download. Do not commit the binaries to ordinary Git history.
- [x] Add the approved author order and contribution roles plus `CITATION.cff`.
- [ ] Add affiliation details and the preferred paper DOI/identifier when author-approved and available.
- [ ] Re-run tests and at least one dataset/model preflight on the released tag in the supported server environment, then archive the exact environment freeze and provenance report.

The historical data root was independently recovered and matched the 22,516 original image hashes and three COCO annotation hashes. The nine schema-v7 manifests and generated labels were also validated against that recovered tree. The server-side checkpoint check above verifies recorded files at that point in time; public-download integrity remains pending.
