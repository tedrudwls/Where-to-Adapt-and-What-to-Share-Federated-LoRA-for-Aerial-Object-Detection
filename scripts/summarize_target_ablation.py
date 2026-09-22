#!/usr/bin/env python3
"""Build strict publication tables for the 3x3x3 LoRA target ablation.

The study is intentionally fixed to the primary AOD-4 protocol:

* methods: FL LoRA, FedSA-LoRA, and Fixed Share-B LoRA;
* targets: decoder-only, backbone-only, and both targets;
* paired training/partition seeds: 42, 43, and 44;
* Dirichlet alpha 0.4, LoRA rank 8, and LoRA alpha 16.

Only the 27 canonical result paths are read.  Unrelated experiments under the
same results root therefore cannot silently enter the ablation table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


METHODS = ("lora", "fedsa_lora", "fixed_share_b_lora")
TARGETS = ("decoder_only", "backbone_only", "both")
SEEDS = (42, 43, 44)

METHOD_LABELS = {
    "lora": "FL LoRA",
    "fedsa_lora": "Share-A/Local-B (FedSA)",
    "fixed_share_b_lora": "Share-B/Local-A (Fixed-B)",
}
TARGET_LABELS = {
    "decoder_only": "decoder-only",
    "backbone_only": "backbone-only",
    "both": "both-targets",
}
BASE_EXPERIMENT_NAMES = {
    "lora": "fl_lora_r8_a0.4",
    "fedsa_lora": "fl_fedsa_lora_r8_a0.4",
    "fixed_share_b_lora": "fl_fixed_share_b_lora_r8_a0.4",
}
TARGET_FLAGS = {
    "decoder_only": (False, True),
    "backbone_only": (True, False),
    "both": (True, True),
}

UTILITY_FIELDS = (
    "macro_AP",
    "macro_AP50",
    "macro_AP75",
    "client_sd_AP",
    "client_sd_AP50",
    "client_sd_AP75",
    "worst_AP",
    "worst_AP50",
    "worst_AP75",
    "common_AP",
    "common_AP50",
    "common_AP75",
)
EFFICIENCY_FIELDS = (
    "trainable_params_m",
    "communication_params_m",
    "one_client_one_way_mb",
    "system_round_total_mb",
    "cumulative_total_mb",
    "parameter_saving_pct",
    "communication_byte_saving_pct",
)
SUMMARY_FIELDS = UTILITY_FIELDS + EFFICIENCY_FIELDS

HIGHER_IS_BETTER = {
    "macro_AP",
    "macro_AP50",
    "macro_AP75",
    "worst_AP",
    "worst_AP50",
    "worst_AP75",
    "common_AP",
    "common_AP50",
    "common_AP75",
    "parameter_saving_pct",
    "communication_byte_saving_pct",
}
LOWER_IS_BETTER = set(SUMMARY_FIELDS) - HIGHER_IS_BETTER

EXPECTED_TRAINING_PROTOCOL = {
    "model_name": "rtdetr-l",
    "num_clients": 3,
    "fl_rounds": 20,
    "local_epochs": 5,
    "partition": "dirichlet",
    "dirichlet_alpha": 0.4,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.0,
    "backbone_min_channels": 64,
    "batch_size": 8,
    "img_size": 640,
    "lr": 0.0003,
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
    "cross_client_eval": True,
    "optimizer": "AdamW",
    "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
    "client_participation": "all_clients_every_round",
    "aggregation_weighting": "local_train_image_count",
    "nonfloating_state_policy": "retain_previous_server_value",
    "validation_frequency_rounds": 1,
    "selection_criterion": "macro_client_local_validation_AP",
    "local_epoch_budget_per_client": 100,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _integer(value, label: str) -> int:
    number = _finite_number(value, label)
    integer = int(number)
    if number != integer:
        raise ValueError(f"{label} must be an integer: {value!r}")
    return integer


def _equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool):
            return False
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _expect(mapping: Mapping, key: str, expected, context: str) -> None:
    if key not in mapping:
        raise ValueError(f"{context} is missing {key!r}")
    actual = mapping[key]
    if not _equal(actual, expected):
        raise ValueError(
            f"{context}.{key} mismatch: actual={actual!r}, expected={expected!r}"
        )


def experiment_name(method: str, target: str) -> str:
    base = BASE_EXPERIMENT_NAMES[method]
    return base if target == "both" else f"{base}_{target}"


def expected_result_path(results_root: Path, method: str, target: str, seed: int) -> Path:
    return (
        results_root
        / f"seed_{seed}"
        / experiment_name(method, target)
        / "fl_results.json"
    )


def _metric_record(payload: Mapping, metric: str, path: Path) -> tuple[float, float, float]:
    client_summary = payload.get("client_summary")
    if not isinstance(client_summary, Mapping):
        raise ValueError(f"{path}: client_summary is missing or invalid")
    record = client_summary.get(metric)
    if not isinstance(record, Mapping):
        raise ValueError(f"{path}: client_summary.{metric} is missing or invalid")
    macro = _finite_number(record.get("macro_mean"), f"{path}: macro {metric}")
    client_sd_raw = record.get("client_sample_sd", record.get("sample_std"))
    client_sd = _finite_number(client_sd_raw, f"{path}: client SD {metric}")
    worst_raw = record.get("worst_client", record.get("worst"))
    worst = _finite_number(worst_raw, f"{path}: worst {metric}")
    for label, value in (("macro", macro), ("client SD", client_sd), ("worst", worst)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}: {label} {metric} is outside [0, 1]: {value}")
    return macro, client_sd, worst


def _common_metric(payload: Mapping, metric: str, path: Path) -> float:
    common = payload.get("common_test")
    if not isinstance(common, Mapping):
        raise ValueError(f"{path}: common_test is missing or invalid")
    value = common.get(metric)
    if value is None:
        summary = common.get("summary_across_personalized_models")
        record = summary.get(metric) if isinstance(summary, Mapping) else None
        value = record.get("macro_mean") if isinstance(record, Mapping) else None
    number = _finite_number(value, f"{path}: common {metric}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{path}: common {metric} is outside [0, 1]: {number}")
    return number


def _communication_metric(payload: Mapping, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _validate_split_metadata(payload: Mapping, path: Path) -> None:
    metadata = payload.get("split_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{path}: split_metadata is missing or invalid")
    expected = {
        "schema_version": 7,
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "num_clients": 3,
        "source_split_policy": "official_aod4_v6",
        "official_split_preserved": True,
        "client_partition_unit": "source_group",
    }
    for key, value in expected.items():
        _expect(metadata, key, value, f"{path}: split_metadata")


def load_run(
    results_root: Path,
    split_dir: Path,
    method: str,
    target: str,
    seed: int,
) -> dict:
    """Load and strictly normalize one canonical ablation result."""
    path = expected_result_path(results_root, method, target, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical target-ablation result: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: result root is not an object")

    _expect(payload, "status", "complete", str(path))
    if _integer(payload.get("result_schema_version"), f"{path}: schema") < 2:
        raise ValueError(f"{path}: result_schema_version must be at least 2")
    top_level_expected = {
        "mode": "fl",
        "fl_method": method,
        "seed": seed,
        "partition_seed": seed,
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "apply_lora_backbone": TARGET_FLAGS[target][0],
        "apply_lora_decoder": TARGET_FLAGS[target][1],
        "num_clients": 3,
        "rounds_planned": 20,
        "rounds_executed": 20,
        "local_epochs": 5,
    }
    for key, value in top_level_expected.items():
        _expect(payload, key, value, str(path))

    expected_method_label = f"FL+{method}"
    _expect(payload, "method", expected_method_label, str(path))

    training = payload.get("training_experiment")
    if not isinstance(training, Mapping):
        raise ValueError(f"{path}: training_experiment is missing or invalid")
    for key, value in EXPECTED_TRAINING_PROTOCOL.items():
        _expect(training, key, value, f"{path}: training_experiment")
    _expect(training, "fl_method", method, f"{path}: training_experiment")
    _expect(training, "seed", seed, f"{path}: training_experiment")
    _expect(training, "partition_seed", seed, f"{path}: training_experiment")
    _expect(
        training,
        "apply_lora_backbone",
        TARGET_FLAGS[target][0],
        f"{path}: training_experiment",
    )
    _expect(
        training,
        "apply_lora_decoder",
        TARGET_FLAGS[target][1],
        f"{path}: training_experiment",
    )

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(f"{path}: architecture is missing or invalid")
    _expect(architecture, "model_name", "rtdetr-l", f"{path}: architecture")
    _expect(architecture, "fine_tuning_mode", method, f"{path}: architecture")
    _expect(
        architecture,
        "ultralytics_version",
        "8.4.126",
        f"{path}: architecture",
    )
    weight_sha256 = architecture.get("model_weight_sha256")
    if not (
        isinstance(weight_sha256, str)
        and len(weight_sha256) == 64
        and all(character in "0123456789abcdef" for character in weight_sha256.lower())
    ):
        raise ValueError(f"{path}: invalid architecture.model_weight_sha256")

    split_file = split_dir / f"split_official_v6_dirichlet_a0.4_c3_s{seed}.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing immutable split manifest: {split_file}")
    recorded_split_digest = payload.get(
        "split_manifest_sha256", payload.get("split_file_sha256")
    )
    actual_split_digest = _file_sha256(split_file)
    if not isinstance(recorded_split_digest, str) or (
        recorded_split_digest.strip().lower() != actual_split_digest
    ):
        raise ValueError(
            f"{path}: split digest mismatch; result={recorded_split_digest!r}, "
            f"actual={actual_split_digest}"
        )
    _validate_split_metadata(payload, path)

    row = {
        "path": str(path.resolve()),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "target": target,
        "target_label": TARGET_LABELS[target],
        "seed": seed,
        "partition_seed": seed,
        "split_manifest_sha256": actual_split_digest,
        "model_weight_sha256": weight_sha256.lower(),
    }
    for metric in ("AP", "AP50", "AP75"):
        macro, client_sd, worst = _metric_record(payload, metric, path)
        row[f"macro_{metric}"] = macro
        row[f"client_sd_{metric}"] = client_sd
        row[f"worst_{metric}"] = worst
        row[f"common_{metric}"] = _common_metric(payload, metric, path)

    parameter_counts = payload.get("parameter_counts")
    communication = payload.get("communication")
    if not isinstance(parameter_counts, Mapping):
        raise ValueError(f"{path}: parameter_counts is missing or invalid")
    if not isinstance(communication, Mapping):
        raise ValueError(f"{path}: communication is missing or invalid")

    trainable_params = _integer(
        parameter_counts.get("trainable_params"), f"{path}: trainable_params"
    )
    communication_params = _integer(
        parameter_counts.get("communication_params"),
        f"{path}: communication_params",
    )
    one_way_params = _integer(
        _communication_metric(communication, "one_client_one_way", "params"),
        f"{path}: one-client one-way params",
    )
    if communication_params != one_way_params:
        raise ValueError(
            f"{path}: parameter_counts.communication_params ({communication_params}) "
            f"does not equal the transmitted payload ({one_way_params})"
        )

    one_way_mb = _finite_number(
        _communication_metric(communication, "one_client_one_way", "mb"),
        f"{path}: one-way MB",
    )
    round_mb = _finite_number(
        _communication_metric(communication, "round_total", "mb"),
        f"{path}: round MB",
    )
    cumulative_mb = _finite_number(
        _communication_metric(communication, "cumulative_total", "mb"),
        f"{path}: cumulative MB",
    )
    if not math.isclose(round_mb, one_way_mb * 2 * 3, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{path}: round communication arithmetic is inconsistent")
    if not math.isclose(cumulative_mb, round_mb * 20, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{path}: cumulative communication arithmetic is inconsistent")
    _expect(communication, "unit", "decimal_MB_1e6_bytes", f"{path}: communication")
    _expect(
        communication,
        "scope",
        "model_tensor_payload_only",
        f"{path}: communication",
    )
    _expect(communication, "num_clients", 3, f"{path}: communication")
    _expect(communication, "rounds_executed", 20, f"{path}: communication")

    row.update(
        {
            "trainable_params_m": trainable_params / 1_000_000.0,
            "communication_params_m": communication_params / 1_000_000.0,
            "one_client_one_way_mb": one_way_mb,
            "system_round_total_mb": round_mb,
            "cumulative_total_mb": cumulative_mb,
            "parameter_saving_pct": _finite_number(
                parameter_counts.get("parameter_saving_pct"),
                f"{path}: parameter saving",
            ),
            "communication_byte_saving_pct": _finite_number(
                communication.get(
                    "byte_saving_vs_full_ft_pct",
                    communication.get("saving_vs_full_ft_pct"),
                ),
                f"{path}: communication byte saving",
            ),
        }
    )
    for field in EFFICIENCY_FIELDS:
        if row[field] < 0.0:
            raise ValueError(f"{path}: {field} must be non-negative")
    for field in ("parameter_saving_pct", "communication_byte_saving_pct"):
        if row[field] > 100.0:
            raise ValueError(f"{path}: {field} is above 100%")
    return row


def collect_runs(results_root: Path, split_dir: Path) -> list[dict]:
    rows = [
        load_run(results_root, split_dir, method, target, seed)
        for method in METHODS
        for target in TARGETS
        for seed in SEEDS
    ]
    if len(rows) != 27:
        raise RuntimeError(f"Selected {len(rows)} runs instead of exactly 27")
    keys = {(row["method"], row["target"], row["seed"]) for row in rows}
    if len(keys) != 27:
        raise RuntimeError("Duplicate method/target/seed observations were selected")

    model_digests = {row["model_weight_sha256"] for row in rows}
    if len(model_digests) != 1:
        raise ValueError(
            "The 27 runs did not use one immutable pretrained checkpoint: "
            f"{sorted(model_digests)}"
        )
    for seed in SEEDS:
        split_digests = {
            row["split_manifest_sha256"] for row in rows if row["seed"] == seed
        }
        if len(split_digests) != 1:
            raise ValueError(
                f"Seed {seed} methods/targets used different split manifests: "
                f"{sorted(split_digests)}"
            )
    return rows


def _mean_sample_sd(values: Iterable[float]) -> tuple[float, float, int]:
    data = [_finite_number(value, "summary value") for value in values]
    if not data:
        raise ValueError("Cannot summarize an empty sample")
    return (
        statistics.fmean(data),
        statistics.stdev(data) if len(data) > 1 else 0.0,
        len(data),
    )


def summarize_runs(rows: Sequence[Mapping]) -> list[dict]:
    index = {(row["method"], row["target"], row["seed"]): row for row in rows}
    summaries = []
    for method in METHODS:
        for target in TARGETS:
            group = [index[(method, target, seed)] for seed in SEEDS]
            summary = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "target": target,
                "target_label": TARGET_LABELS[target],
                "runs": len(group),
                "seeds": ",".join(str(seed) for seed in SEEDS),
            }
            for field in SUMMARY_FIELDS:
                mean, sample_sd, count = _mean_sample_sd(row[field] for row in group)
                summary[f"{field}_mean"] = mean
                summary[f"{field}_run_sd"] = sample_sd
                summary[f"{field}_n"] = count
            summaries.append(summary)
    return summaries


def paired_deltas(rows: Sequence[Mapping]) -> list[dict]:
    """Return seed-paired target and policy contrasts with directional wins."""
    index = {(row["method"], row["target"], row["seed"]): row for row in rows}
    contrasts = []

    target_contrasts = (
        ("both_minus_decoder", "both", "decoder_only"),
        ("both_minus_backbone", "both", "backbone_only"),
        ("backbone_minus_decoder", "backbone_only", "decoder_only"),
    )
    for method in METHODS:
        for name, numerator, denominator in target_contrasts:
            contrasts.append(
                ("target", METHOD_LABELS[method], name, method, numerator, method, denominator)
            )

    policy_contrasts = (
        ("fixed_b_minus_fedsa", "fixed_share_b_lora", "fedsa_lora"),
        ("fixed_b_minus_fl_lora", "fixed_share_b_lora", "lora"),
        ("fedsa_minus_fl_lora", "fedsa_lora", "lora"),
    )
    for target in TARGETS:
        for name, numerator, denominator in policy_contrasts:
            contrasts.append(
                (
                    "policy",
                    TARGET_LABELS[target],
                    name,
                    numerator,
                    target,
                    denominator,
                    target,
                )
            )

    output = []
    for axis, context, name, num_method, num_target, den_method, den_target in contrasts:
        for metric in SUMMARY_FIELDS:
            deltas = [
                index[(num_method, num_target, seed)][metric]
                - index[(den_method, den_target, seed)][metric]
                for seed in SEEDS
            ]
            mean, sample_sd, count = _mean_sample_sd(deltas)
            direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
            tolerance = 1e-12
            if direction == "higher":
                wins = sum(delta > tolerance for delta in deltas)
            else:
                wins = sum(delta < -tolerance for delta in deltas)
            ties = sum(abs(delta) <= tolerance for delta in deltas)
            output.append(
                {
                    "comparison_axis": axis,
                    "context": context,
                    "comparison": name,
                    "numerator_method": num_method,
                    "numerator_target": num_target,
                    "denominator_method": den_method,
                    "denominator_target": den_target,
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_runs": count,
                    "mean_delta": mean,
                    "run_sd": sample_sd,
                    "wins_for_numerator": wins,
                    "ties": ties,
                    **{
                        f"seed_{seed}_delta": delta
                        for seed, delta in zip(SEEDS, deltas)
                    },
                }
            )
    return output


def _atomic_write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _format_pm(row: Mapping, field: str, digits: int = 4) -> str:
    return (
        f"{row[f'{field}_mean']:.{digits}f} "
        f"± {row[f'{field}_run_sd']:.{digits}f}"
    )


def markdown_tables(summaries: Sequence[Mapping]) -> str:
    lines = [
        "# RT-DETR LoRA Target-Placement Ablation",
        "",
        "All entries are paired-run mean ± run sample SD over training/partition ",
        "seed pairs (42,42), (43,43), and (44,44). `Client SD` is first computed ",
        "across the three clients within each run and is then summarized across seeds.",
        "",
        "## Client-local performance",
        "",
        "| Policy | Target | Macro AP | AP50 | AP75 | Client SD AP | "
        "Client SD AP50 | Client SD AP75 | Worst AP | Worst AP50 | Worst AP75 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method_label']} | {row['target_label']} | "
            f"{_format_pm(row, 'macro_AP')} | {_format_pm(row, 'macro_AP50')} | "
            f"{_format_pm(row, 'macro_AP75')} | {_format_pm(row, 'client_sd_AP')} | "
            f"{_format_pm(row, 'client_sd_AP50')} | "
            f"{_format_pm(row, 'client_sd_AP75')} | {_format_pm(row, 'worst_AP')} | "
            f"{_format_pm(row, 'worst_AP50')} | {_format_pm(row, 'worst_AP75')} |"
        )
    lines.extend(
        [
            "",
            "## Common pooled-test performance",
            "",
            "| Policy | Target | Common AP | Common AP50 | Common AP75 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['method_label']} | {row['target_label']} | "
            f"{_format_pm(row, 'common_AP')} | {_format_pm(row, 'common_AP50')} | "
            f"{_format_pm(row, 'common_AP75')} |"
        )
    lines.extend(
        [
            "",
            "## Parameter and communication efficiency",
            "",
            "Communication uses decimal MB and counts all three client uploads and "
            "post-aggregation downloads in each round.",
            "",
            "| Policy | Target | Trainable M | Communication M | One-way MB | "
            "Round MB | Total MB | Parameter saving (%) | Communication saving (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['method_label']} | {row['target_label']} | "
            f"{_format_pm(row, 'trainable_params_m', 6)} | "
            f"{_format_pm(row, 'communication_params_m', 6)} | "
            f"{_format_pm(row, 'one_client_one_way_mb', 6)} | "
            f"{_format_pm(row, 'system_round_total_mb', 6)} | "
            f"{_format_pm(row, 'cumulative_total_mb', 6)} | "
            f"{_format_pm(row, 'parameter_saving_pct', 3)} | "
            f"{_format_pm(row, 'communication_byte_saving_pct', 3)} |"
        )
    lines.extend(
        [
            "",
            "## Paired-delta interpretation",
            "",
            "The companion CSV reports numerator-minus-denominator differences at "
            "identical seeds. A numerator win means a positive delta for higher-is-better "
            "metrics and a negative delta for lower-is-better metrics. With only three "
            "paired seeds, win counts and effect sizes are descriptive rather than "
            "confirmatory significance tests.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    runs: Sequence[Mapping],
    summaries: Sequence[Mapping],
    deltas: Sequence[Mapping],
) -> dict[str, Path]:
    run_fields = (
        "method",
        "method_label",
        "target",
        "target_label",
        "seed",
        "partition_seed",
        "split_manifest_sha256",
        "model_weight_sha256",
        *SUMMARY_FIELDS,
        "path",
    )
    summary_fields = [
        "method",
        "method_label",
        "target",
        "target_label",
        "runs",
        "seeds",
    ]
    for field in SUMMARY_FIELDS:
        summary_fields.extend(
            (f"{field}_mean", f"{field}_run_sd", f"{field}_n")
        )
    delta_fields = list(deltas[0]) if deltas else []

    paths = {
        "runs": output_dir / "target_ablation_runs.csv",
        "summary": output_dir / "target_ablation_summary.csv",
        "deltas": output_dir / "target_ablation_paired_deltas.csv",
        "markdown": output_dir / "target_ablation_tables.md",
    }
    _atomic_write_csv(paths["runs"], runs, run_fields)
    _atomic_write_csv(paths["summary"], summaries, summary_fields)
    _atomic_write_csv(paths["deltas"], deltas, delta_fields)
    _atomic_write_text(paths["markdown"], markdown_tables(summaries))
    return paths


def run(results_root: Path, split_dir: Path, output_dir: Path) -> dict[str, Path]:
    runs = collect_runs(results_root.resolve(), split_dir.resolve())
    summaries = summarize_runs(runs)
    deltas = paired_deltas(runs)
    return write_outputs(output_dir.resolve(), runs, summaries, deltas)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly summarize the 27 primary r8/alpha0.4 FL LoRA "
            "target-placement ablation runs."
        )
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        default="./results/official_v6",
        help="Root containing seed_42, seed_43, and seed_44 result directories",
    )
    parser.add_argument(
        "--split_dir",
        default="./data/splits",
        help="Directory containing the three immutable official-v6 split manifests",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output directory; defaults to "
            "<results_root>/summary_target_ablation_r8_a0.4"
        ),
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).expanduser()
    split_dir = Path(args.split_dir).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else results_root / "summary_target_ablation_r8_a0.4"
    )
    try:
        paths = run(results_root, split_dir, output_dir)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"[INVALID] {error}")
        return 1

    print("[PASS] Exactly 27 protocol-matched target-ablation runs were summarized")
    for label, path in paths.items():
        print(f"[{label.upper()}] {path}")
    print()
    print(paths["markdown"].read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
