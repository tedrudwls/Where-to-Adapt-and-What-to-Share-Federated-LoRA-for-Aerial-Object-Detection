# Security and membership-inference scope

The experiment does not transmit raw images or annotations between simulated clients. This is a **data-locality property**, not proof of confidentiality: shared model updates can contain information about training data. No differential-privacy mechanism, server-update attack, encrypted transport or deployment network trace was evaluated.

The endpoint membership-inference analysis is a **label-aware loss-threshold attack** against the selected model, using source-disjoint member/nonmember attack samples. AUC has a 0.5 chance baseline; balanced attack accuracy (ASR) likewise has 0.5 chance. The score direction and thresholds are chosen on a calibration partition and evaluated on a held-out partition. A lower attack AUC under this one threat model is an empirical observation, not a universal privacy ranking.

The frozen three-seed robustness audit additionally reports `initial_loss` (fresh-initialization negative control) and `delta_loss = trained_loss - initial_loss` (nonstandard training-change diagnostic). The later covariate-matched analysis was motivated by above-chance initial-loss AUC in the unmatched audit and therefore is **exploratory post-hoc** evidence, not preregistered confirmation. It matches background, capped per-class instance composition, object count and mean normalized box-area bins after source-group splitting; unobserved scene/sensor confounding can remain.

| FL policy | Covariate-matched trained-loss AUC | Matched balanced ASR |
| --- | ---: | ---: |
| FedAvg full fine-tuning | 0.5735 ± 0.0038 | 0.5529 ± 0.0028 |
| FedLoRA-AB | 0.5446 ± 0.0034 | 0.5275 ± 0.0040 |
| FedLoRA-A | 0.5779 ± 0.0072 | 0.5530 ± 0.0036 |
| FedLoRA-B | 0.5661 ± 0.0270 | 0.5427 ± 0.0183 |

These are means ± sample SD over **three paired training/partition seeds**; repeated attack splits and clients are nested measurements. Values below 0.5 are treated as attack instability, not “extra privacy.” The low-FPR operating points require reporting **achieved** evaluation FPR, and the covariate-matched local audit omitted the 1% point because its matched nonmember partitions could not consistently resolve that threshold.

The audit code is included under `scripts/`; historical per-image caches and audit outputs are not yet public in this source tree. The conclusions should not claim a formal privacy guarantee, a server-side leakage result, or security of a deployed military system.
