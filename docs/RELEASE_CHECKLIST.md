# Publication release gate

This checklist separates **available source/provenance** from **still-private binary artifacts**. A draft pull request does not mean that the paper artifact is complete.

- [x] Curate experiment source, tests, launchers, data manifest provenance and documentation without raw AOD-4 images.
- [x] Record all 96 selected checkpoint names, sizes, SHA-256 digests and prerequisite pretrained/split digests.
- [x] Verify seven critical executable source files against the archived server code snapshot. The `requirements.txt` package pins match, but its explanatory Python-version comment was corrected for this release draft, so its file SHA differs. The remaining helpers still need a final server-vs-release comparison before tagging a reproducibility release.
- [x] Compile Python, syntax-check shell scripts, run dependency-free tests and validate README/doc links in the local draft.
- [ ] Choose the exact project code license and document separate terms for learned checkpoints, original RT-DETR-L weights and the AOD-4 dataset. Do not infer permission for one from permission for another.
- [ ] On the server, run `python3 scripts/verify_checkpoint_assets.py --project-dir /home/gpuadmin/kim/fedsalora` against all 96 binaries and confirm `rtdetr-l.pt` SHA-256.
- [ ] Recheck that the primary result JSONs, frozen split manifests and selected checkpoint hashes did not change during publication preparation.
- [ ] Add and validate a separate read-only checkpoint evaluation CLI on the original server and a safe data-root relocation protocol. Do not use `main.py --resume` to evaluate archived primary results because it writes result files.
- [ ] Publish the 96 uniquely named checkpoint binaries and original result JSON bundle as versioned, checksummed assets, then verify every public download and link the immutable release tag in the README. The locally received 96-JSON archive digest is recorded in `artifacts/result_bundle_sha256.txt`; it is not yet a public download. Do not commit the binaries to ordinary Git history.
- [ ] Add approved authors, institutional/affiliation details, manuscript DOI/identifier if available, and `CITATION.cff`.
- [ ] Re-run tests and at least one dataset/model preflight on the released tag in the supported server environment, then archive the exact environment freeze and provenance report.

The historical data root was independently recovered and matched the 22,516 original image hashes and three COCO annotation hashes. The nine schema-v7 manifests and generated labels were also validated against that recovered tree. These checks do **not** replace the pending checkpoint-binary and public-download checks.
