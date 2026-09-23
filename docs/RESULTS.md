# Results and metric definitions

The tables below describe the **recorded** AOD-4 v6 experiments, not new runs produced by this release candidate. Every AP value here is multiplied by 100 relative to the 0–1 values in the result JSONs. The replicate unit is the paired `(training seed, partition seed)` tuple `(42,42)`, `(43,43)`, or `(44,44)`; `±` is sample standard deviation across those three runs. Three clients within a run are **not** three independent replicates.

## Primary comparison: Dirichlet α=0.4, rank 8, both LoRA targets

| Method | Client-macro AP | Client-macro AP50 | Client-macro AP75 | Within-run client SD (AP) | Worst-client AP | Common-test AP | Total model-tensor payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg full fine-tuning | 67.51 ± 3.82 | 95.00 ± 2.98 | 74.92 ± 3.99 | 6.52 ± 5.10 | 60.61 ± 10.15 | 67.46 ± 0.41 | 15,783.24 MB |
| FedLoRA-AB | 65.39 ± 3.90 | 94.46 ± 3.09 | 72.26 ± 4.45 | 6.28 ± 5.38 | 58.71 ± 10.50 | 65.41 ± 0.68 | 212.78 MB |
| FedLoRA-A (local B) | 60.58 ± 6.67 | 88.70 ± 7.94 | 66.42 ± 8.31 | 7.03 ± 4.46 | 53.74 ± 11.46 | 53.57 ± 5.49 | 131.68 MB |
| FedLoRA-B (local A) | 60.76 ± 7.29 | 89.77 ± 7.50 | 66.57 ± 9.02 | 6.17 ± 3.35 | 54.51 ± 10.76 | 56.29 ± 4.00 | 85.05 MB |

The full model is the accuracy reference; AB is the strongest of the evaluated rank-8 **LoRA** sharing policies. Relative to full fine-tuning, AB is 2.12 client-macro AP points lower and accounts for 98.65% fewer model-state bytes. A and B reduce payload further but show larger common-test AP loss, especially under stronger label skew. There is no claim that any method dominates on every metric or every seed.

## What each AP statistic measures

- **Client AP:** AP of a client's aligned model on that client's held-out test partition. A global method uses the same global model for each partition; a personalized method uses its client-specific factors.
- **Client-macro AP:** arithmetic mean of three client AP values within a run, then mean and sample SD over three runs. It gives each client equal weight, independent of test-partition image count.
- **Within-run client SD:** sample SD of the three client AP values, then mean and sample SD of that statistic across runs. A lower value is not, by itself, evidence of better worst-client AP.
- **Worst-client AP:** minimum of the three client AP values within each run, then mean and sample SD over runs.
- **Common-test AP:** each model is independently evaluated against the *same entire official test set*. For a personalized method, the three resulting AP values are averaged; this is **not** an ensemble or a single pooled prediction set.
- **Common-test per-class AP:** class-specific AP on the full official common test set.
- **Client-local per-class AP:** class-specific AP of one aligned client model on that client's complete local test partition. The calculation includes every image in that partition, not only images positive for the class, so false positives on negative/background images affect AP. Class `support` records the number of ground-truth boxes; it is not the number of images used to calculate AP.

Checkpoint selection used validation AP (macro client-local validation AP for FL) before the test evaluation. Neither test AP nor MIA results selected the reported checkpoint.

## Placement and heterogeneity

At rank 8, client-macro AP for decoder-only / backbone-only / both was:

| Sharing policy | Decoder only | Backbone only | Both |
| --- | ---: | ---: | ---: |
| FedLoRA-AB | 55.41 | 65.11 | 65.39 |
| FedLoRA-A | 53.62 | 59.90 | 60.58 |
| FedLoRA-B | 53.64 | 60.03 | 60.76 |

The AB backbone-only configuration has 64.85 common-test AP and 165.59 MB payload, versus 65.41 and 212.78 MB for both targets. The target configurations do not hold trainable parameter count fixed, so this is a configuration comparison, not an estimate of per-parameter causal value.

| Policy | IID common-test AP | Dirichlet α=0.1 common-test AP | Seed-aligned IID→α=0.1 drop |
| --- | ---: | ---: | ---: |
| FedAvg full fine-tuning | 67.69 | 66.26 | 1.43 ± 0.45 |
| FedLoRA-AB | 65.81 | 63.67 | 2.13 ± 1.82 |
| FedLoRA-A | 63.24 | 48.76 | 14.48 ± 1.24 |
| FedLoRA-B | 63.87 | 52.61 | 11.25 ± 1.08 |

Each distribution regime uses its own deterministic partition manifest. Accordingly, the drop measures a protocol-level sensitivity across paired seeds, not an identical-client, matched-image counterfactual. The α=0.4, α=0.1 and IID experiments all use the same official train/validation/test membership; only the client assignments change.

## Payload accounting

Payload is an **analytical model-tensor-state** count from the in-process FL simulator, not a network trace. The accounting includes three clients × 20 rounds × one upload and one post-aggregation download of the shared tensors. It excludes the initial common state `G⁰`, the final validation-selected checkpoint redistribution, transport/serialization headers, encryption, compression, and raw images. Full fine-tuning contains FP32 state plus INT64 counters; LoRA policies communicate selected trainable tensors and the shared task-head state. Decimal MB means `10⁶` bytes.

| Policy | Trainable parameters | Shared communication values | Per-round payload | 20-round payload |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning | 32.8143 M | 32.8817 M values | 789.16 MB | 15,783.24 MB |
| FedLoRA-AB | 0.4433 M | 0.4433 M | 10.64 MB | 212.78 MB |
| FedLoRA-A | 0.4433 M | 0.2743 M | 6.58 MB | 131.68 MB |
| FedLoRA-B | 0.4433 M | 0.1772 M | 4.25 MB | 85.05 MB |

The full-FT exact shared-state count is **32,881,740 values**, which rounds to **32.8817 M**, not 32.8816 M. Its value count is not interchangeable with its byte count because some buffers are INT64. `score_head` and `class_embed` together account for 8,220 trainable task-head parameters; they are shared under all three LoRA FL policies. Consult the archived per-run JSON and dtype audit for exact integer byte counts.

## Scope of this release

The checkpoint index covers 96 selected models: 36 primary, 18 placement-only, 18 rank-4/16, and 24 IID/α=0.1. Shared rank-8 both-target runs appear in multiple analyses but are counted only once. The public result JSON bundle and executable aggregation are pending binary-artifact verification; until then, tables here should be treated as **reported values** with the stated provenance, not as freshly recomputed numbers from files in this source tree.
