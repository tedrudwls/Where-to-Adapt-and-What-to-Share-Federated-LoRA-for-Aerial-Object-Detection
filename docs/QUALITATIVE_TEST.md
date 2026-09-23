# Held-out qualitative illustrations: seed-43 zero-local-support case

The two figures in the README are static, read-only illustrations from historical validation-selected checkpoints. They are not a new training run, an AP re-evaluation, or evidence selected from model outcomes. The path-sanitized selection, checkpoint, prediction, and figure records are preserved in [`artifacts/qualitative_unseen_helicopter_s43_c1.json`](../artifacts/qualitative_unseen_helicopter_s43_c1.json). No additional case-specific rendering utility is distributed for these committed figures.

## Case and quantitative context

- Partition: Dirichlet alpha 0.4, paired training/partition seed 43, three clients.
- Focal endpoint: client 1, rank-8 Backbone+Decoder LoRA.
- Client 1 local training support for helicopter: 0 boxes in 0 images.
- Client 1 validation support: 142 boxes in 142 positive images. Validation performance selected the checkpoint, so this is not a zero-shot or open-vocabulary setting.
- Client 1 local test partition: 748 total images, including 75 helicopter boxes in 75 positive images.
- Other-client training support: 5,528 helicopter boxes.

Helicopter AP below is client-local per-class AP on the **complete 748-image local test partition**, not AP restricted to the 75 positive images. Negative/background images and false positives therefore contribute to the metric.

| Sharing policy | Client-local helicopter AP (0–100) |
| --- | ---: |
| FedLoRA-AB | 62.91 |
| FedLoRA-A (Share-A / local B) | 0.09 |
| FedLoRA-B (Share-B / local A) | 8.03 |

## Outcome-blind image selection

The selection was frozen before inference using only the official test annotations and the audited source-group identity:

1. Select client 1 test images containing a helicopter box.
2. Require the image's Roboflow-filename/SHA-256 connected source group to be absent from both training and validation.
3. Sort the resulting 75 candidates by normalized helicopter ground-truth box area, using image ID to make rank ties deterministic.
4. Freeze the p10, p50, and p90 cases. The README displays p10 and p50 for a compact, legible presentation; omitting p90 from the display does not affect any quantitative result.

| Stratum | Official-test image | Area rank | Normalized GT area | README display |
| --- | ---: | ---: | ---: | --- |
| p10 / small | 62 | 8 / 75 | 0.0026178360 | Yes |
| p50 / median | 1114 | 38 / 75 | 0.0351476669 | Yes |
| p90 / large | 1177 | 68 / 75 | 0.4103660583 | No |

The source-disjoint statement applies to these selected examples only. It does not make the full official test split source-disjoint and does not alter the official AP tables.

## Inference record

All methods use the corresponding rank-8 validation-selected personalized endpoint, 640-pixel inference, and a fixed display confidence threshold of 0.25. The threshold controls only which boxes appear in the figures; COCO AP is computed independently of this display threshold. Selected rounds differ because validation selected each endpoint separately.

| Method | Experiment | Selected round | Selected-checkpoint SHA-256 |
| --- | --- | ---: | --- |
| FedLoRA-AB | `seed_43/fl_lora_r8_a0.4` | 20 | `ad928c84c5f1a82f7f88334ce5825422d1249bd59cb934eaeb3ddac3a751ff02` |
| FedLoRA-A | `seed_43/fl_fedsa_lora_r8_a0.4` | 20 | `c0ebf42091e3381c933a824c0b53f993af72c016c2e5015105ef90889448c616` |
| FedLoRA-B | `seed_43/fl_fixed_share_b_lora_r8_a0.4` | 6 | `f454435dc0a37530c919ccfff921ddfbe84fb18bc1919477cad186fdf513116d` |

For image 62, FedLoRA-AB labels the target as helicopter at 0.578 and also produces an off-target drone false positive at 0.287. FedLoRA-A labels the target as drone at 0.608 and retains another drone false positive at 0.425; FedLoRA-B labels the target as drone at 0.686. For image 1114, FedLoRA-AB labels the target as helicopter at 0.875, while FedLoRA-A and FedLoRA-B label it as drone at 0.821 and 0.669. The committed figures retain these displayed outputs rather than suppressing unfavorable predictions.

## Interpretation boundary

- This is one naturally occurring seed/client/class case, not a repeated leave-one-class-out experiment. It has no run SD, confidence interval, or significance claim.
- The figures illustrate model behavior and are consistent with the full-partition class AP; two images cannot establish aggregate or causal superiority.
- All three LoRA policies share classification/denoising head state, begin from a pretrained representation, and use helicopter-positive validation data for checkpoint selection. The result cannot isolate a causal semantic role for A or B alone.
- Federated updates can carry helicopter-relevant information from the other clients, but raw images and boxes are not transmitted by the evaluated FL protocol.
- The public JSON removes historical server paths while retaining split, image, source-group, checkpoint, result, prediction, and figure hashes needed to audit the committed record.
