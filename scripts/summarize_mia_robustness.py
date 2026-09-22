#!/usr/bin/env python3
"""Aggregate the frozen seed-42/43/44 read-only MIA audits.

One statistical replicate is one paired ``(training seed, partition seed)``
audit.  Attack-split repeats and clients are never promoted to independent
replicates.  The script reads completed audit reports and integrity manifests,
then writes only to a dedicated sibling summary directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # Package import in tests; direct sibling import for `python scripts/...`.
    from scripts.mia_robustness_audit import EXPECTED_AUGMENTATION_PROTOCOL
except ModuleNotFoundError:  # pragma: no cover - exercised by direct server CLI
    from mia_robustness_audit import EXPECTED_AUGMENTATION_PROTOCOL


SEEDS = (42, 43, 44)
METHODS = ("full_ft", "lora", "fedsa_lora", "fixed_share_b_lora")
EXPERIMENT_NAMES = {
    "full_ft": "fl_full_ft_a0.4",
    "lora": "fl_lora_r8_a0.4",
    "fedsa_lora": "fl_fedsa_lora_r8_a0.4",
    "fixed_share_b_lora": "fl_fixed_share_b_lora_r8_a0.4",
}
SCOPES = ("local", "pooled")
SCORES = ("trained_loss", "initial_loss", "delta_loss")
METRICS = (
    "auc_roc",
    "asr",
    "tpr_at_calibration_target_1fpr_threshold",
    "evaluation_fpr_at_calibration_target_1fpr_threshold",
    "tpr_at_calibration_target_5fpr_threshold",
    "evaluation_fpr_at_calibration_target_5fpr_threshold",
    "tpr_at_calibration_target_10fpr_threshold",
    "evaluation_fpr_at_calibration_target_10fpr_threshold",
)
FROZEN_PROTOCOL = {
    "selection_seed": 420042,
    "attack_seed": 842042,
    "attack_repeats": 20,
    "calibration_fraction": 0.5,
    "max_member_samples": 1000,
    "max_local_nonmember_samples": 1000,
    "max_pooled_nonmember_samples": 2000,
    "calibration_evaluation_partition_unit": "source_group",
}
PASS_GATES = (
    "read_only_primary_gate",
    "exact_initial_digest_gate",
    "checkpoint_primary_architecture_manifest_gate",
    "post_extraction_initial_model_state_gate",
    "sample_pairing_across_methods_gate",
    "repeated_attack_plan_pairing_gate",
)


def audit_directory_name(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise ValueError(f"Expected one of the frozen audit seeds {SEEDS}; received {seed}")
    return "security_audit_v1" if int(seed) == 42 else f"security_audit_v1_seed{seed}"


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _atomic_json(payload, path: Path) -> None:
    _atomic_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        path,
    )


def _atomic_csv(rows: Sequence[Mapping], fields: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _require_probability(value, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite probability; received {value}")
    return value


def _validate_integrity_manifest(manifest: dict, report_path: Path) -> dict:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported integrity schema for {report_path}")
    if manifest.get("status") != "pass" or manifest.get("read_only_primary_gate") is not True:
        raise ValueError(f"Integrity gate did not pass for {report_path}")
    if manifest.get("changes") != []:
        raise ValueError(f"Integrity manifest records protected-file changes for {report_path}")
    before, after = manifest.get("before"), manifest.get("after")
    if not isinstance(before, dict) or not before or before != after:
        raise ValueError(f"Integrity before/after snapshots disagree for {report_path}")
    if (
        manifest.get("generated_yolo_tree_sha256_before")
        != manifest.get("generated_yolo_tree_sha256_after")
    ):
        raise ValueError(f"Generated YOLO tree changed during {report_path}")
    for recorded_path, record in before.items():
        path = Path(recorded_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Protected audit input is missing or unsafe: {path}")
        if int(record.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Protected audit input size changed after audit: {path}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Protected audit input hash changed after audit: {path}")
    records = manifest.get("audit_outputs", {})
    report_records = [
        row for row in records.values()
        if isinstance(row, dict) and Path(str(row.get("path", ""))).name == "audit_report.json"
    ]
    if len(report_records) != 1:
        raise ValueError(f"Missing unique audit_report.json integrity record for {report_path}")
    if report_records[0].get("sha256") != sha256_file(report_path):
        raise ValueError(f"Audit report changed after its integrity manifest: {report_path}")
    return before


def _primary_result_path(snapshot: Mapping[str, dict], *, method: str,
                         seed: int) -> Path:
    suffix = f"/seed_{seed}/{EXPERIMENT_NAMES[method]}/fl_results.json"
    matches = [Path(path) for path in snapshot if str(path).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one protected primary result for seed={seed}, method={method}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_primary_training_protocol(payload: dict, *, method: str,
                                        seed: int, path: Path) -> str:
    uses_lora = method != "full_ft"
    training = payload.get("training_experiment")
    if not isinstance(training, dict):
        raise ValueError(f"Missing training_experiment in {path}")
    expected = {
        "fl_method": method,
        "model_name": "rtdetr-l",
        "num_clients": 3,
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "fl_rounds": 20,
        "local_epochs": 5,
        "batch_size": 8,
        "img_size": 640,
        "lr": 0.0003 if uses_lora else 0.0001,
        "head_lr": 0.0001,
        "backbone_lr_ratio": 0.1,
        "weight_decay": 0.0001,
        "warmup_epochs": 5.0,
        "min_lr_ratio": 0.01,
        "grad_clip_norm": 0.1,
        "close_mosaic_epochs": 10,
        "fedprox_mu": 0.0,
        "reset_optimizer_each_round": True,
        "amp": False,
        "patience": 0,
        "val_interval": 5,
        "seed": int(seed),
        "partition_seed": int(seed),
        "num_workers": 4,
        "cross_client_eval": True,
        "optimizer": "AdamW",
        "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
        "selection_criterion": "macro_client_local_validation_AP",
        "augmentation_protocol": EXPECTED_AUGMENTATION_PROTOCOL,
    }
    if uses_lora:
        expected.update({
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.0,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
            "backbone_min_channels": 64,
        })
    top_expected = {
        "status": "complete",
        "mode": "fl",
        "fl_method": method,
        "seed": int(seed),
        "partition_seed": int(seed),
        "partition": "dirichlet",
        "num_clients": 3,
        "rounds_executed": 20,
        "local_epochs": 5,
    }
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"Missing architecture manifest in {path}")
    architecture_expected = {
        "model_name": "rtdetr-l",
        "num_classes": 4,
        "class_names": ["airplane", "bird", "drone", "helicopter"],
    }
    mismatches = {
        f"result.{key}": {"observed": payload.get(key), "expected": value}
        for key, value in top_expected.items() if payload.get(key) != value
    }
    mismatches.update({
        f"training_experiment.{key}": {
            "observed": training.get(key), "expected": value,
        }
        for key, value in expected.items() if training.get(key) != value
    })
    mismatches.update({
        f"architecture.{key}": {
            "observed": architecture.get(key), "expected": value,
        }
        for key, value in architecture_expected.items()
        if architecture.get(key) != value
    })
    if uses_lora:
        top_lora = {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
        }
        mismatches.update({
            f"result.{key}": {"observed": payload.get(key), "expected": value}
            for key, value in top_lora.items() if payload.get(key) != value
        })
    if mismatches:
        raise ValueError(
            f"Primary training protocol mismatch in {path}: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )
    normalized = dict(training)
    for key in ("seed", "partition_seed", "model_weights"):
        normalized.pop(key, None)
    return json_sha256(normalized)


def validate_report(report: dict, *, seed: int, report_path: Path) -> None:
    if int(report.get("audit_schema_version", -1)) != 1:
        raise ValueError(f"Unsupported audit schema in {report_path}")
    if report.get("status") != "complete":
        raise ValueError(f"Incomplete audit report: {report_path}")
    if report.get("audit_name") != "security_audit_v1":
        raise ValueError(f"Unexpected audit name in {report_path}")
    if int(report.get("seed", -1)) != int(seed):
        raise ValueError(f"Audit seed mismatch in {report_path}")
    # Seed 42 was produced before audit_instance was added; accept only its
    # absence or the exact backward-compatible name.
    instance = report.get("audit_instance")
    expected_instance = audit_directory_name(seed)
    if (seed == 42 and instance not in (None, expected_instance)) or (
        seed != 42 and instance != expected_instance
    ):
        raise ValueError(f"Audit instance mismatch in {report_path}: {instance!r}")
    for gate in PASS_GATES:
        if report.get(gate) != "pass":
            raise ValueError(f"Required gate {gate!r} did not pass in {report_path}")
    if seed != 42 and report.get("frozen_primary_training_protocol_gate") != "pass":
        raise ValueError(f"Frozen primary-training gate did not pass in {report_path}")
    if report.get("protocol") != FROZEN_PROTOCOL:
        raise ValueError(f"Frozen protocol mismatch in {report_path}")
    if set(report.get("methods", {})) != set(METHODS):
        raise ValueError(f"Expected exactly the four paired methods in {report_path}")

    for method in METHODS:
        macro = report["methods"][method].get("macro", {})
        for scope in SCOPES:
            for score in SCORES:
                for metric in METRICS:
                    try:
                        cell = macro[scope][score][metric]
                        mean = cell["client_macro_mean"]
                        client_sd = cell["client_sample_sd"]
                    except (KeyError, TypeError) as error:
                        raise ValueError(
                            f"Missing {method}/{scope}/{score}/{metric} in {report_path}"
                        ) from error
                    _require_probability(mean, f"{method}/{scope}/{score}/{metric}/mean")
                    _require_probability(
                        client_sd, f"{method}/{scope}/{score}/{metric}/client_sd"
                    )


def load_reports(results_root: os.PathLike | str,
                 seeds: Sequence[int] = SEEDS) -> tuple[dict[int, dict], list[dict]]:
    root = Path(results_root).resolve(strict=True)
    if tuple(int(seed) for seed in seeds) != SEEDS:
        raise ValueError(f"Frozen multiseed summary requires exactly seeds {SEEDS}")
    reports: dict[int, dict] = {}
    provenance = []
    for seed in SEEDS:
        directory = root / audit_directory_name(seed)
        report_path = (directory / "audit_report.json").resolve(strict=True)
        integrity_path = (directory / "integrity_manifest.json").resolve(strict=True)
        report = _read_json(report_path)
        integrity = _read_json(integrity_path)
        protected_snapshot = _validate_integrity_manifest(integrity, report_path)
        validate_report(report, seed=seed, report_path=report_path)
        primary_protocols = {}
        for method in METHODS:
            primary_path = _primary_result_path(
                protected_snapshot, method=method, seed=seed
            )
            primary_payload = _read_json(primary_path)
            primary_protocols[method] = _validate_primary_training_protocol(
                primary_payload, method=method, seed=seed, path=primary_path
            )
        reports[seed] = report
        provenance.append({
            "seed": seed,
            "audit_directory": str(directory.resolve(strict=True)),
            "audit_report_sha256": sha256_file(report_path),
            "integrity_manifest_sha256": sha256_file(integrity_path),
            "split_manifest_sha256": report["split_manifest_sha256"],
            "model_weight_sha256": report["model_weight_sha256"],
            "primary_training_manifest_sha256_by_method": primary_protocols,
            "checkpoint_compatibility_protocol_sha256_by_method": {
                method: report["methods"][method].get(
                    "primary_training_protocol_sha256"
                )
                for method in METHODS
            },
        })
    model_hashes = {row["model_weight_sha256"] for row in provenance}
    if len(model_hashes) != 1:
        raise ValueError("The three audits did not use the same pretrained model weights")
    for method in METHODS:
        fingerprints = {
            row["primary_training_manifest_sha256_by_method"][method]
            for row in provenance
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"Primary training protocol drift across seeds for method={method}: "
                f"{sorted(fingerprints)}"
            )
        new_audit_fingerprints = {
            row["checkpoint_compatibility_protocol_sha256_by_method"][method]
            for row in provenance if row["seed"] in (43, 44)
        }
        if (
            None in new_audit_fingerprints
            or any(not isinstance(value, str) or len(value) != 64
                   for value in new_audit_fingerprints)
            or len(new_audit_fingerprints) != 1
        ):
            raise ValueError(
                "Checkpoint compatibility-protocol fingerprint mismatch across "
                f"new audits for method={method}: {sorted(map(str, new_audit_fingerprints))}"
            )
    return reports, provenance


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("At least two independent paired replicates are required")
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def aggregate_reports(reports: Mapping[int, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if tuple(sorted(int(seed) for seed in reports)) != SEEDS:
        raise ValueError(f"Expected reports for exactly seeds {SEEDS}")
    long_rows = []
    values_by_key: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for method in METHODS:
        for scope in SCOPES:
            for score in SCORES:
                for metric in METRICS:
                    values = {
                        seed: _require_probability(
                            reports[seed]["methods"][method]["macro"][scope][score]
                            [metric]["client_macro_mean"],
                            f"seed={seed}/{method}/{scope}/{score}/{metric}",
                        )
                        for seed in SEEDS
                    }
                    client_sds = {
                        seed: _require_probability(
                            reports[seed]["methods"][method]["macro"][scope][score]
                            [metric]["client_sample_sd"],
                            f"seed={seed}/{method}/{scope}/{score}/{metric}/client_sd",
                        )
                        for seed in SEEDS
                    }
                    mean, replicate_sd = _mean_sd(values.values())
                    client_sd_mean, client_sd_replicate_sd = _mean_sd(client_sds.values())
                    row = {
                        "method": method,
                        "scope": scope,
                        "score": score,
                        "metric": metric,
                        "n_paired_replicates": len(SEEDS),
                        "mean": mean,
                        "replicate_sample_sd": replicate_sd,
                        "mean_within_replicate_client_sd": client_sd_mean,
                        "within_replicate_client_sd_replicate_sample_sd": client_sd_replicate_sd,
                        **{f"seed_{seed}": values[seed] for seed in SEEDS},
                    }
                    long_rows.append(row)
                    values_by_key[(method, scope, score, metric)] = values

    method_deltas = []
    for first, second in combinations(METHODS, 2):
        for scope in SCOPES:
            for score in SCORES:
                for metric in METRICS:
                    deltas = {
                        seed: values_by_key[(first, scope, score, metric)][seed]
                        - values_by_key[(second, scope, score, metric)][seed]
                        for seed in SEEDS
                    }
                    mean, replicate_sd = _mean_sd(deltas.values())
                    method_deltas.append({
                        "first_method": first,
                        "second_method": second,
                        "difference": "first_minus_second",
                        "scope": scope,
                        "score": score,
                        "metric": metric,
                        "n_paired_replicates": len(SEEDS),
                        "mean_difference": mean,
                        "paired_replicate_sample_sd": replicate_sd,
                        **{f"seed_{seed}_difference": deltas[seed] for seed in SEEDS},
                    })

    score_deltas = []
    for method in METHODS:
        for scope in SCOPES:
            for metric in METRICS:
                deltas = {
                    seed: values_by_key[(method, scope, "trained_loss", metric)][seed]
                    - values_by_key[(method, scope, "delta_loss", metric)][seed]
                    for seed in SEEDS
                }
                mean, replicate_sd = _mean_sd(deltas.values())
                score_deltas.append({
                    "method": method,
                    "scope": scope,
                    "metric": metric,
                    "difference": "trained_loss_minus_delta_loss",
                    "n_paired_replicates": len(SEEDS),
                    "mean_difference": mean,
                    "paired_replicate_sample_sd": replicate_sd,
                    **{f"seed_{seed}_difference": deltas[seed] for seed in SEEDS},
                })
    return long_rows, method_deltas, score_deltas


def _index(rows: Sequence[dict]) -> dict[tuple[str, str, str, str], dict]:
    return {
        (row["method"], row["scope"], row["score"], row["metric"]): row
        for row in rows
    }


def _pm(row: Mapping) -> str:
    return f"{float(row['mean']):.4f} ± {float(row['replicate_sample_sd']):.4f}"


def _operating_cell(index: Mapping, method: str, scope: str, score: str,
                    target: int) -> str:
    tpr = index[(method, scope, score,
                 f"tpr_at_calibration_target_{target}fpr_threshold")]
    fpr = index[(method, scope, score,
                 f"evaluation_fpr_at_calibration_target_{target}fpr_threshold")]
    return f"{_pm(tpr)} / {_pm(fpr)}"


def build_markdown(rows: Sequence[dict]) -> str:
    index = _index(rows)
    lines = [
        "# Frozen-protocol MIA robustness summary (seeds 42/43/44)",
        "",
        "Values are mean $\\pm$ sample SD over three paired replicates "
        "`(training seed, partition seed) = (42,42), (43,43), (44,44)`. "
        "Clients and the 20 repeated attack splits are not treated as independent "
        "replicates. AUC and balanced accuracy have a 0.5 chance baseline; values near "
        "0.5 indicate that this attack is uninformative and below-chance noise must not "
        "be ranked as extra privacy. TPR is interpreted together with achieved FPR. "
        "These are attack-specific observations, not a formal privacy guarantee.",
        "",
        "`ASR` below means balanced attack accuracy on a held-out balanced evaluation "
        "set after score direction and the Youden-J threshold are selected only on the "
        "calibration subset. Each "
        "low-FPR cell is `evaluation TPR / achieved evaluation FPR`; 1%, 5%, and 10% "
        "refer to the calibration-targeted FPR threshold.",
    ]

    def attack_table(title: str, scope: str, score: str) -> None:
        lines.extend([
            "", f"## {title}", "",
            "| Method | AUC | Balanced attack accuracy (ASR) | TPR/FPR @1% | TPR/FPR @5% | TPR/FPR @10% |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            lines.append(
                f"| {method} | {_pm(index[(method, scope, score, 'auc_roc')])} | "
                f"{_pm(index[(method, scope, score, 'asr')])} | "
                f"{_operating_cell(index, method, scope, score, 1)} | "
                f"{_operating_cell(index, method, scope, score, 5)} | "
                f"{_operating_cell(index, method, scope, score, 10)} |"
            )

    attack_table("Primary local trained-loss MIA", "local", "trained_loss")
    attack_table(
        "Primary local initialization-referenced loss-change diagnostic",
        "local", "delta_loss",
    )

    lines.extend([
        "",
        "> The delta-loss table is a sensitivity/diagnostic analysis, not a standardized "
        "FL-MIA and not a differential-privacy result.",
        "",
        "## Local fresh-initialization negative control",
        "",
        "| Method | Initial-loss AUC | Initial-loss balanced attack accuracy |",
        "|---|---:|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {method} | {_pm(index[(method, 'local', 'initial_loss', 'auc_roc')])} | "
            f"{_pm(index[(method, 'local', 'initial_loss', 'asr')])} |"
        )

    lines.extend([
        "",
        "## Secondary pooled-test sensitivity view",
        "",
        "| Method | Trained-loss AUC | Trained-loss ASR | Delta-loss AUC | Delta-loss ASR |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {method} | {_pm(index[(method, 'pooled', 'trained_loss', 'auc_roc')])} | "
            f"{_pm(index[(method, 'pooled', 'trained_loss', 'asr')])} | "
            f"{_pm(index[(method, 'pooled', 'delta_loss', 'auc_roc')])} | "
            f"{_pm(index[(method, 'pooled', 'delta_loss', 'asr')])} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- `trained_loss` is a conventional label-aware final-checkpoint loss-threshold MIA score.",
        "- `delta_loss = trained_loss - initial_loss` is an initialization-referenced "
        "loss-change diagnostic, not LiRA and not a standardized calibrated MIA.",
        "- The endpoint audit restores personalized local factors. It does not model the "
        "information visible to an honest-but-curious server from client updates.",
        "- With only three paired replicates, report effect sizes, paired directions and "
        "sample SD; do not claim formal significance or privacy guarantees.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        parser.error(f"the frozen summary requires --seeds {' '.join(map(str, SEEDS))}")
    return args


def main(argv=None) -> int:
    os.umask(0o077)
    args = _parse_args(argv)
    root = Path(args.results_root).resolve(strict=True)
    expected_output = root / "security_audit_v1_multiseed_summary"
    output = Path(args.output_dir).resolve(strict=False) if args.output_dir else expected_output
    if output != expected_output:
        raise ValueError(f"Summary output must be the dedicated sibling {expected_output}")
    if output.is_symlink():
        raise ValueError(f"Refusing symbolic-link output directory: {output}")
    if output.exists():
        symlinks = [str(path) for path in output.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError("Summary output contains unsafe symbolic links: " + ", ".join(symlinks[:10]))
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)

    reports, provenance = load_reports(root, args.seeds)
    rows, method_deltas, score_deltas = aggregate_reports(reports)
    common_fields = [
        "method", "scope", "score", "metric", "n_paired_replicates", "mean",
        "replicate_sample_sd", "mean_within_replicate_client_sd",
        "within_replicate_client_sd_replicate_sample_sd",
        "seed_42", "seed_43", "seed_44",
    ]
    method_delta_fields = [
        "first_method", "second_method", "difference", "scope", "score", "metric",
        "n_paired_replicates", "mean_difference", "paired_replicate_sample_sd",
        "seed_42_difference", "seed_43_difference", "seed_44_difference",
    ]
    score_delta_fields = [
        "method", "scope", "metric", "difference", "n_paired_replicates",
        "mean_difference", "paired_replicate_sample_sd", "seed_42_difference",
        "seed_43_difference", "seed_44_difference",
    ]
    _atomic_csv(rows, common_fields, output / "summary_long.csv")
    _atomic_csv(method_deltas, method_delta_fields, output / "paired_method_differences.csv")
    _atomic_csv(score_deltas, score_delta_fields, output / "paired_trained_delta_differences.csv")
    _atomic_text(build_markdown(rows), output / "summary.md")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "audit_name": "security_audit_v1_multiseed_summary",
        "replicate_definition": "paired (training seed, partition seed); one audit per seed",
        "seeds": list(SEEDS),
        "n_paired_replicates": len(SEEDS),
        "protocol": FROZEN_PROTOCOL,
        "protocol_sha256": json_sha256(FROZEN_PROTOCOL),
        "statistical_unit": "paired training/partition-seed audit",
        "non_independent_units": ["client", "repeated attack calibration/evaluation split"],
        "provenance": provenance,
        "rows": rows,
        "paired_method_differences": method_deltas,
        "paired_trained_delta_differences": score_deltas,
        "interpretation": (
            "Attack-specific empirical membership robustness; not a formal privacy, "
            "confidentiality, or differential-privacy guarantee."
        ),
    }
    _atomic_json(summary, output / "summary.json")
    generated = ("summary_long.csv", "paired_method_differences.csv",
                 "paired_trained_delta_differences.csv", "summary.md", "summary.json")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_audits": provenance,
        "outputs": {
            name: {"sha256": sha256_file(output / name),
                   "size_bytes": (output / name).stat().st_size}
            for name in generated
        },
    }
    _atomic_json(manifest, output / "summary_manifest.json")
    print(f"[PASS] Aggregated {len(SEEDS)} paired frozen-protocol MIA audits")
    print(f"[Summary] {output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
