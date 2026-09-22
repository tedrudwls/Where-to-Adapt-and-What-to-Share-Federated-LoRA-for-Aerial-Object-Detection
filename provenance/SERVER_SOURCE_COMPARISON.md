# Server-source comparison for the release draft

The authors supplied a source-only archive named `source_code_ney1s8Yt.tar.gz`, SHA-256 `294d75c7209d8c3118a433268555ef1430bff0a74cfb0c7a0d77ea34bc272f3c`. The archive was inspected without executing its contents. It contains 60 regular files and six directory entries; no unsafe paths, duplicate members, symbolic links, `__pycache__` or `.pyc` entries were found.

Of the **54 code/configuration files shared** by the server archive and this GitHub draft, **49 are byte-identical**. Five have intentional release-only edits:

| File | Server SHA-256 | Draft SHA-256 | Scope of edit |
| --- | --- | --- | --- |
| `requirements.txt` | `9b05fd88d5aeec0b1d368217f70d784669895e935c8edb7cd4591d26c665eea4` | `faa7ddd7253eac016d58263e4b26f2004d585e3e4c35e5eeec4ea2d34b7da11a` | Corrected the explanatory recorded-Python-version comment; dependency pins unchanged. |
| `scripts/launch_heterogeneity_24_tmux.sh` | `47348accd82a5f31ccb7fc77cff7d57deab2e857af1f89a41d626751cc11a6bb` | `ca515a4128aa9cf058a81bbf3695392cf2ae6691cac5361d79570aa2c0915b0b` | Replaced fixed server paths with environment-configurable paths. |
| `scripts/prepare_heterogeneity_splits.sh` | `a2db7007662ead0948c3980620fb9a6acc73d528f40423c474597bf2b8394c30` | `423ffc42075441520e6f43dc6f5fcdc3fb38fcc095ca9078bfcdccd8f7109a8c` | Replaced fixed server paths with environment-configurable paths. |
| `scripts/run_heterogeneity_queue.sh` | `eaa85383f05e84777fb884342e8435003e00a993bdcc6f73954998c94b154c56` | `1b46d6fb01b6a62e6eacdcc09e0137a566f71b1291ee3b03e2a1ce3520e69fb2` | Replaced fixed server paths with environment-configurable paths. |
| `scripts/verify_heterogeneity_24.sh` | `5694d20ed2988587415ef7491bf40af22edf3f982cf7d7b09b01c10c7347af3a` | `d6bce6299bd41437dd4cc1389c04b525eeb30394b93996ced14d0fb0a60fcb80` | Replaced fixed server paths with environment-configurable paths. |

The reviewed diffs do **not** change model architecture, learning schedule, partition algorithm, aggregation or evaluation semantics. Four draft-only files were added for publication (`scripts/run_lora_grid.sh`, `scripts/verify_checkpoint_assets.py`, `tests/test_verify_checkpoint_assets.py`, `scripts/make_paper_figures.py`); they were not part of the supplied server archive and must not be represented as historical execution code.

The server archive also contains four **recovery/evidence-only** scripts not used as training entry points: `scripts/extract_aod4_official_zip.py`, `scripts/verify_aod4_recovery.py`, `scripts/validate_recovered_aod4_manifests.py`, and `scripts/collect_submission_evidence.sh`. They are intentionally excluded from this core source draft until their historical-cache/path dependencies are made portable and their generated evidence is reviewed for server paths, package URLs and per-image attack records. The two `.DS_Store` entries were excluded as operating-system metadata. Do not redistribute the entire server evidence bundle without a separate disclosure review.
