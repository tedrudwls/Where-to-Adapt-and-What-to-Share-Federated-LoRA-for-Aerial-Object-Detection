#!/usr/bin/env python3
"""Summarize the naturally occurring client-local zero-support class case.

This analysis is fixed to the completed primary AOD-4 experiment family:

* paired training/partition seeds 42, 43, and 44;
* three clients, official AOD-4 v6 membership, and Dirichlet alpha 0.4;
* eight primary methods under the rank-eight joint-target configuration.

The script first discovers zero-positive-support cells using only the immutable
split manifests (not performance), then validates all 36 canonical primary
result files.  It reports the sole post-hoc descriptive case selected by this
data-only rule on the common pooled test set.  Primary results and split
manifests are hashed before and after analysis and are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SEEDS = (42, 43, 44)
NUM_CLIENTS = 3
EXPECTED_ALPHA = 0.4
EXPECTED_RANK = 8
EXPECTED_LORA_ALPHA = 16.0
EXPECTED_CLASS = "helicopter"
EXPECTED_CASE = (43, 1, EXPECTED_CLASS)
EXPECTED_COMMON_TEST_SUPPORT = 787
EXPECTED_CLASS_NAMES = ("airplane", "bird", "drone", "helicopter")
EXPECTED_CATEGORY_MAP = {"1": 0, "2": 1, "3": 2, "4": 3}

EXPECTED_COMMON_TRAINING_PROTOCOL = {
    "model_name": "rtdetr-l",
    "num_clients": NUM_CLIENTS,
    "partition": "dirichlet",
    "dirichlet_alpha": EXPECTED_ALPHA,
    "batch_size": 8,
    "img_size": 640,
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
    "cross_client_eval": True,
    "optimizer": "AdamW",
    "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    mode: str
    fl_method: str
    experiment: str
    result_name: str
    client_specific_file: bool
    model_scope: str
    information_path: str
    baseline_key: str | None


METHODS = (
    MethodSpec(
        "local_full_ft",
        "Local Full FT",
        "solo",
        "full_ft",
        "solo_full_ft_client{client_id}",
        "solo_results.json",
        True,
        "focal-client local model",
        "no cross-client training path",
        None,
    ),
    MethodSpec(
        "local_lora",
        "Local LoRA",
        "solo",
        "lora",
        "solo_lora_client{client_id}",
        "solo_results.json",
        True,
        "focal-client local model",
        "no cross-client training path",
        None,
    ),
    MethodSpec(
        "centralized_full_ft",
        "Centralized Full FT",
        "centralized",
        "full_ft",
        "centralized_full_ft",
        "centralized_results.json",
        False,
        "single pooled-data model",
        "raw-data pooling (contextual reference)",
        "local_full_ft",
    ),
    MethodSpec(
        "centralized_lora",
        "Centralized LoRA",
        "centralized",
        "lora",
        "centralized_lora",
        "centralized_results.json",
        False,
        "single pooled-data model",
        "raw-data pooling (contextual reference)",
        "local_lora",
    ),
    MethodSpec(
        "fl_full_ft",
        "FL Full FT",
        "fl",
        "full_ft",
        "fl_full_ft_a0.4",
        "fl_results.json",
        False,
        "shared/global federated model",
        "aggregated full model state (no raw boxes/images)",
        "local_full_ft",
    ),
    MethodSpec(
        "fl_lora",
        "FL LoRA",
        "fl",
        "lora",
        "fl_lora_r8_a0.4",
        "fl_results.json",
        False,
        "shared/global federated model",
        "aggregated LoRA A + LoRA B + task head (no raw boxes/images)",
        "local_lora",
    ),
    MethodSpec(
        "fedsa_lora",
        "FedSA-LoRA",
        "fl",
        "fedsa_lora",
        "fl_fedsa_lora_r8_a0.4",
        "fl_results.json",
        False,
        "personalized federated endpoint",
        "aggregated shared A + task head; local B (no raw boxes/images)",
        "local_lora",
    ),
    MethodSpec(
        "fixed_share_b_lora",
        "Fixed Share-B",
        "fl",
        "fixed_share_b_lora",
        "fl_fixed_share_b_lora_r8_a0.4",
        "fl_results.json",
        False,
        "personalized federated endpoint",
        "aggregated shared B + task head; local A (no raw boxes/images)",
        "local_lora",
    ),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_with_sha256(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload, digest


def _finite(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _integer(value, label: str) -> int:
    number = _finite(value, label)
    integer = int(number)
    if number != integer:
        raise ValueError(f"{label} must be an integer: {value!r}")
    return integer


def _equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
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


def split_path(split_dir: Path, seed: int) -> Path:
    return split_dir / f"split_official_v6_dirichlet_a0.4_c3_s{seed}.json"


def result_path(
    results_root: Path, spec: MethodSpec, seed: int, client_id: int | None = None
) -> Path:
    if spec.client_specific_file:
        if client_id is None:
            raise ValueError(f"{spec.key} requires a client ID")
        experiment = spec.experiment.format(client_id=client_id)
    else:
        experiment = spec.experiment
    return results_root / f"seed_{seed}" / experiment / spec.result_name


def _manifest_class_map(metadata: Mapping, path: Path) -> tuple[list[str], dict[str, int]]:
    class_names = metadata.get("class_names")
    raw_mapping = metadata.get("cat_id_to_label")
    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError(f"{path}: metadata.class_names is invalid")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(f"{path}: metadata.cat_id_to_label is invalid")
    if tuple(class_names) != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"{path}: unexpected class order {class_names!r}; "
            f"expected {list(EXPECTED_CLASS_NAMES)!r}"
        )
    mapping = {
        str(raw_id): _integer(label, f"{path}: category mapping {raw_id}")
        for raw_id, label in raw_mapping.items()
    }
    if mapping != EXPECTED_CATEGORY_MAP:
        raise ValueError(
            f"{path}: unexpected category-to-label mapping {mapping!r}; "
            f"expected {EXPECTED_CATEGORY_MAP!r}"
        )
    return class_names, mapping


def _client_by_id(manifest: Mapping, client_id: int, path: Path) -> Mapping:
    clients = manifest.get("clients")
    if not isinstance(clients, list):
        raise ValueError(f"{path}: clients is missing or invalid")
    matches = [
        client
        for client in clients
        if isinstance(client, Mapping)
        and _integer(client.get("client_id"), f"{path}: client ID") == client_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one client {client_id}")
    return matches[0]


def _support(
    client: Mapping, split: str, raw_category_id: str, field: str, path: Path
) -> int:
    splits = client.get("splits")
    block = splits.get(split) if isinstance(splits, Mapping) else None
    values = block.get(field) if isinstance(block, Mapping) else None
    if not isinstance(values, Mapping) or raw_category_id not in values:
        raise ValueError(
            f"{path}: client split {split}.{field}.{raw_category_id} is missing"
        )
    value = _integer(
        values[raw_category_id],
        f"{path}: client split {split}.{field}.{raw_category_id}",
    )
    if value < 0:
        raise ValueError(f"{path}: class support cannot be negative")
    return value


def load_split_manifests(split_dir: Path) -> tuple[dict[int, dict], dict[Path, str]]:
    manifests: dict[int, dict] = {}
    hashes: dict[Path, str] = {}
    for seed in SEEDS:
        path = split_path(split_dir, seed)
        manifest, digest = _load_json_with_sha256(path)
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}: metadata is missing or invalid")
        expected = {
            "schema_version": 7,
            "partition": "dirichlet",
            "dirichlet_alpha": EXPECTED_ALPHA,
            "num_clients": NUM_CLIENTS,
            "seed": seed,
            "source_split_policy": "official_aod4_v6",
            "official_split_preserved": True,
            "client_partition_unit": "source_group",
            "min_bbox_area": 0.0,
            "min_bbox_side": 0.0,
            "drop_empty_images": False,
            "crowd_policy": "require_zero_crowd_annotations_for_YOLO_metric_equivalence",
            "category_policy": (
                "exact_aod4_targets_ignore_only_unreferenced_declared_categories"
            ),
        }
        for key, value in expected.items():
            _expect(metadata, key, value, f"{path}: metadata")
        _manifest_class_map(metadata, path)
        clients = manifest.get("clients")
        if not isinstance(clients, list) or len(clients) != NUM_CLIENTS or not all(
            isinstance(client, Mapping) for client in clients
        ):
            raise ValueError(f"{path}: clients must contain exactly three objects")
        client_ids = sorted(
            _integer(client.get("client_id"), f"{path}: client ID")
            for client in clients
        )
        if client_ids != list(range(NUM_CLIENTS)):
            raise ValueError(f"{path}: client IDs must be 0, 1, and 2")
        manifests[seed] = manifest
        hashes[path] = digest
    return manifests, hashes


def discover_zero_support_cases(manifests: Mapping[int, Mapping]) -> list[dict]:
    cases: list[dict] = []
    for seed in SEEDS:
        manifest = manifests[seed]
        path = Path(f"split(seed={seed})")
        metadata = manifest["metadata"]
        class_names, mapping = _manifest_class_map(metadata, path)
        for client_id in range(NUM_CLIENTS):
            client = _client_by_id(manifest, client_id, path)
            for raw_category_id, label_index in sorted(mapping.items()):
                class_name = class_names[label_index]
                train_boxes = _support(
                    client, "train", raw_category_id, "class_instances", path
                )
                train_images = _support(
                    client, "train", raw_category_id, "class_images", path
                )
                if (train_boxes == 0) != (train_images == 0):
                    raise ValueError(
                        f"{path}: inconsistent zero support for client {client_id} "
                        f"class {class_name}"
                    )
                if train_boxes != 0:
                    continue
                other_train_boxes_by_client = {
                    str(other_id): _support(
                        _client_by_id(manifest, other_id, path),
                        "train",
                        raw_category_id,
                        "class_instances",
                        path,
                    )
                    for other_id in range(NUM_CLIENTS)
                    if other_id != client_id
                }
                other_train_boxes = sum(other_train_boxes_by_client.values())
                common_test_boxes = sum(
                    _support(
                        _client_by_id(manifest, other_id, path),
                        "test",
                        raw_category_id,
                        "class_instances",
                        path,
                    )
                    for other_id in range(NUM_CLIENTS)
                )
                cases.append(
                    {
                        "seed": seed,
                        "client_id": client_id,
                        "class_name": class_name,
                        "label_index": label_index,
                        "raw_category_id": raw_category_id,
                        "train_boxes": train_boxes,
                        "train_images": train_images,
                        "val_boxes": _support(
                            client, "val", raw_category_id, "class_instances", path
                        ),
                        "val_images": _support(
                            client, "val", raw_category_id, "class_images", path
                        ),
                        "local_test_boxes": _support(
                            client, "test", raw_category_id, "class_instances", path
                        ),
                        "local_test_images": _support(
                            client, "test", raw_category_id, "class_images", path
                        ),
                        "other_client_train_boxes": other_train_boxes,
                        "other_client_train_boxes_by_client": (
                            other_train_boxes_by_client
                        ),
                        "common_test_boxes": common_test_boxes,
                    }
                )
    return cases


def _stored_split_digest(payload: Mapping, path: Path) -> str:
    values = {
        str(payload[key]).strip().lower()
        for key in ("split_file_sha256", "split_manifest_sha256")
        if payload.get(key) is not None
    }
    if len(values) != 1:
        raise ValueError(f"{path}: missing or inconsistent split SHA-256 fields")
    digest = next(iter(values))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{path}: invalid split SHA-256")
    return digest


def _validate_result(
    payload: Mapping,
    path: Path,
    spec: MethodSpec,
    seed: int,
    expected_split_digest: str,
    expected_split_metadata: Mapping,
    client_id: int | None,
) -> None:
    if _integer(payload.get("result_schema_version"), f"{path}: schema") < 2:
        raise ValueError(f"{path}: result schema must be at least 2")
    expected = {
        "status": "complete",
        "mode": spec.mode,
        "fl_method": spec.fl_method,
        "seed": seed,
        "partition_seed": seed,
        "partition": "dirichlet",
        "dirichlet_alpha": EXPECTED_ALPHA,
        "num_clients": NUM_CLIENTS,
    }
    for key, value in expected.items():
        _expect(payload, key, value, str(path))
    expected_method = (
        f"Local-{spec.fl_method}"
        if spec.mode == "solo"
        else f"Centralized-{spec.fl_method}"
        if spec.mode == "centralized"
        else f"FL+{spec.fl_method}"
    )
    _expect(payload, "method", expected_method, str(path))
    if spec.client_specific_file:
        _expect(payload, "client_id", client_id, str(path))
    if spec.fl_method != "full_ft":
        for key, value in {
            "lora_rank": EXPECTED_RANK,
            "lora_alpha": EXPECTED_LORA_ALPHA,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
        }.items():
            _expect(payload, key, value, str(path))
    if _stored_split_digest(payload, path) != expected_split_digest:
        raise ValueError(f"{path}: result does not match the supplied split manifest")

    training = payload.get("training_experiment")
    if not isinstance(training, Mapping):
        raise ValueError(f"{path}: training_experiment is missing or invalid")
    for key, value in EXPECTED_COMMON_TRAINING_PROTOCOL.items():
        _expect(training, key, value, f"{path}: training_experiment")
    _expect(training, "fl_method", spec.fl_method, f"{path}: training_experiment")
    _expect(training, "seed", seed, f"{path}: training_experiment")
    _expect(training, "partition_seed", seed, f"{path}: training_experiment")
    _expect(
        training,
        "lr",
        0.0001 if spec.fl_method == "full_ft" else 0.0003,
        f"{path}: training_experiment",
    )
    if spec.fl_method != "full_ft":
        for key, value in {
            "lora_rank": EXPECTED_RANK,
            "lora_alpha": EXPECTED_LORA_ALPHA,
            "lora_dropout": 0.0,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
            "backbone_min_channels": 64,
        }.items():
            _expect(training, key, value, f"{path}: training_experiment")
    if spec.mode == "fl":
        for key, value in {
            "fl_rounds": 20,
            "local_epochs": 5,
            "client_participation": "all_clients_every_round",
            "aggregation_weighting": "local_train_image_count",
            "nonfloating_state_policy": "retain_previous_server_value",
            "validation_frequency_rounds": 1,
            "selection_criterion": "macro_client_local_validation_AP",
            "local_epoch_budget_per_client": 100,
        }.items():
            _expect(training, key, value, f"{path}: training_experiment")
        _expect(payload, "rounds_planned", 20, str(path))
        _expect(payload, "rounds_executed", 20, str(path))
        _expect(payload, "local_epochs", 5, str(path))
    else:
        budget_key = "solo_epochs" if spec.mode == "solo" else "centralized_epochs"
        selection = (
            "single_client_validation_AP"
            if spec.mode == "solo"
            else "macro_client_local_validation_AP"
        )
        _expect(training, budget_key, 100, f"{path}: training_experiment")
        _expect(
            training,
            "checkpoint_selection",
            selection,
            f"{path}: training_experiment",
        )
        _expect(payload, "epochs_budget", 100, str(path))
        training_record = payload.get("training")
        if not isinstance(training_record, Mapping):
            raise ValueError(f"{path}: training record is missing or invalid")
        _expect(training_record, "epochs_executed", 100, f"{path}: training")

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(f"{path}: architecture is missing or invalid")
    for key, value in {
        "model_name": "rtdetr-l",
        "num_classes": 4,
        "class_names": list(EXPECTED_CLASS_NAMES),
        "fine_tuning_mode": spec.fl_method,
        "ultralytics_version": "8.4.126",
    }.items():
        _expect(architecture, key, value, f"{path}: architecture")
    weight_digest = str(architecture.get("model_weight_sha256", "")).lower()
    if len(weight_digest) != 64 or any(
        character not in "0123456789abcdef" for character in weight_digest
    ):
        raise ValueError(f"{path}: invalid architecture.model_weight_sha256")

    split_metadata = payload.get("split_metadata")
    if not isinstance(split_metadata, Mapping):
        raise ValueError(f"{path}: split_metadata is missing")
    if split_metadata != expected_split_metadata:
        raise ValueError(
            f"{path}: embedded split_metadata differs from the immutable manifest"
        )
    for key, value in {
        "schema_version": 7,
        "partition": "dirichlet",
        "dirichlet_alpha": EXPECTED_ALPHA,
        "num_clients": NUM_CLIENTS,
        "source_split_policy": "official_aod4_v6",
        "official_split_preserved": True,
        "client_partition_unit": "source_group",
    }.items():
        _expect(split_metadata, key, value, f"{path}: split_metadata")


def collect_primary_results(
    results_root: Path,
    split_dir: Path,
    manifests: Mapping[int, Mapping],
    split_hashes: Mapping[Path, str],
) -> tuple[dict[tuple[int, str, int | None], dict], dict[Path, str]]:
    records: dict[tuple[int, str, int | None], dict] = {}
    input_hashes: dict[Path, str] = dict(split_hashes)
    for seed in SEEDS:
        expected_digest = split_hashes[split_path(split_dir, seed)]
        expected_metadata = manifests[seed]["metadata"]
        for spec in METHODS:
            client_ids: Sequence[int | None] = (
                tuple(range(NUM_CLIENTS)) if spec.client_specific_file else (None,)
            )
            for client_id in client_ids:
                path = result_path(results_root, spec, seed, client_id)
                payload, digest = _load_json_with_sha256(path)
                _validate_result(
                    payload,
                    path,
                    spec,
                    seed,
                    expected_digest,
                    expected_metadata,
                    client_id,
                )
                records[(seed, spec.key, client_id)] = payload
                input_hashes[path] = digest
    if len(records) != 36:
        raise RuntimeError(f"Expected 36 canonical primary results, found {len(records)}")
    weight_digests = {
        str(payload["architecture"]["model_weight_sha256"]).lower()
        for payload in records.values()
    }
    if len(weight_digests) != 1:
        raise ValueError(
            "Canonical primary results do not share one pretrained-weight SHA-256: "
            f"{sorted(weight_digests)}"
        )
    return records, input_hashes


def _validated_metric_record(block: Mapping, class_name: str, context: str) -> dict:
    per_class = block.get("per_class")
    record = per_class.get(class_name) if isinstance(per_class, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"{context}: missing per_class.{class_name}")
    support_map = block.get("class_support")
    support_copy = (
        support_map.get(class_name) if isinstance(support_map, Mapping) else None
    )
    support = _integer(record.get("support"), f"{context}: per-class support")
    if support_copy is None or _integer(
        support_copy, f"{context}: class-support copy"
    ) != support:
        raise ValueError(f"{context}: inconsistent common-test class support")
    metrics = {"support": support}
    for metric in ("AP", "AP50", "AP75"):
        value = _finite(record.get(metric), f"{context}: {metric}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{context}: {metric} is outside [0, 1]: {value}")
        metrics[metric] = value
    return metrics


def _common_class_metrics(
    payload: Mapping, spec: MethodSpec, client_id: int, class_name: str, path: Path
) -> dict:
    common = payload.get("common_test")
    if not isinstance(common, Mapping):
        raise ValueError(f"{path}: common_test is missing")
    if spec.mode != "fl":
        return _validated_metric_record(common, class_name, str(path))

    entries = common.get("per_client_model")
    if (
        not isinstance(entries, list)
        or len(entries) != NUM_CLIENTS
        or not all(isinstance(entry, Mapping) for entry in entries)
    ):
        raise ValueError(
            f"{path}: common_test.per_client_model must contain exactly three objects"
        )
    ids = [
        _integer(entry.get("client_id"), f"{path}: common-test client ID")
        for entry in entries
    ]
    if sorted(ids) != list(range(NUM_CLIENTS)) or len(ids) != NUM_CLIENTS:
        raise ValueError(f"{path}: common-test client model IDs must be exactly 0,1,2")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and _integer(entry.get("client_id"), f"{path}: client ID") == client_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one common-test model for client {client_id}")
    entry = matches[0]
    nested = entry.get("metrics")
    if not isinstance(nested, Mapping):
        raise ValueError(f"{path}: client {client_id} nested metrics are missing")
    nested_metrics = _validated_metric_record(
        nested, class_name, f"{path}: client {client_id} nested metrics"
    )
    if "per_class" in entry or "class_support" in entry:
        direct_metrics = _validated_metric_record(
            entry, class_name, f"{path}: client {client_id} direct metrics"
        )
        if direct_metrics != nested_metrics:
            raise ValueError(f"{path}: direct and nested client metrics disagree")
    return nested_metrics


def _embedded_case_support(
    payload: Mapping, case: Mapping, path: Path
) -> tuple[int, int]:
    metadata = payload.get("split_metadata")
    stats = metadata.get("realized_partition_statistics") if isinstance(metadata, Mapping) else None
    train_rows = stats.get("train") if isinstance(stats, Mapping) else None
    if not isinstance(train_rows, list):
        raise ValueError(f"{path}: embedded realized train statistics are missing")
    matches = [
        row
        for row in train_rows
        if isinstance(row, Mapping)
        and _integer(row.get("client_id"), f"{path}: embedded client ID")
        == case["client_id"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: embedded focal-client statistics are ambiguous")
    row = matches[0]
    raw_id = case["raw_category_id"]
    boxes = row.get("class_instances")
    images = row.get("class_images")
    if not isinstance(boxes, Mapping) or not isinstance(images, Mapping):
        raise ValueError(f"{path}: embedded class support is missing")
    return (
        _integer(boxes.get(raw_id), f"{path}: embedded train boxes"),
        _integer(images.get(raw_id), f"{path}: embedded train images"),
    )


def extract_case_rows(
    results_root: Path,
    records: Mapping[tuple[int, str, int | None], Mapping],
    case: Mapping,
) -> list[dict]:
    seed = int(case["seed"])
    client_id = int(case["client_id"])
    class_name = str(case["class_name"])
    rows: list[dict] = []
    for spec in METHODS:
        record_key = (seed, spec.key, client_id if spec.client_specific_file else None)
        payload = records[record_key]
        path = result_path(
            results_root, spec, seed, client_id if spec.client_specific_file else None
        )
        embedded_boxes, embedded_images = _embedded_case_support(payload, case, path)
        if embedded_boxes != 0 or embedded_images != 0:
            raise ValueError(f"{path}: embedded focal-client train support is not zero")
        metrics = _common_class_metrics(payload, spec, client_id, class_name, path)
        rows.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "model_scope": spec.model_scope,
                "information_path": spec.information_path,
                "seed": seed,
                "client_id": client_id,
                "class_name": class_name,
                "local_train_boxes": int(case["train_boxes"]),
                "local_train_images": int(case["train_images"]),
                "local_val_boxes": int(case["val_boxes"]),
                "local_val_images": int(case["val_images"]),
                "local_test_boxes": int(case["local_test_boxes"]),
                "local_test_images": int(case["local_test_images"]),
                "other_client_train_boxes": int(case["other_client_train_boxes"]),
                "other_client_train_boxes_contributing_signal": (
                    0 if spec.mode == "solo" else int(case["other_client_train_boxes"])
                ),
                "common_test_support": int(metrics["support"]),
                "AP": float(metrics["AP"]),
                "AP50": float(metrics["AP50"]),
                "AP75": float(metrics["AP75"]),
                "baseline_key": spec.baseline_key,
                "result_file": str(path.resolve()),
                "result_file_sha256": _sha256(path),
            }
        )
    support_values = {row["common_test_support"] for row in rows}
    if support_values != {int(case["common_test_boxes"])}:
        raise ValueError(f"Common pooled-test support is inconsistent: {support_values}")
    if support_values != {EXPECTED_COMMON_TEST_SUPPORT}:
        raise ValueError(
            "Official AOD-4 v6 helicopter test support changed: "
            f"actual={support_values}, expected={EXPECTED_COMMON_TEST_SUPPORT}"
        )
    by_key = {row["method_key"]: row for row in rows}
    for row in rows:
        baseline_key = row["baseline_key"]
        row["delta_AP_vs_matched_local"] = (
            None if baseline_key is None else row["AP"] - by_key[baseline_key]["AP"]
        )
    return rows


def build_comparisons(rows: Sequence[Mapping]) -> list[dict]:
    by_key = {row["method_key"]: row for row in rows}
    pairs = (
        ("centralized_full_ft", "local_full_ft"),
        ("centralized_lora", "local_lora"),
        ("fl_full_ft", "local_full_ft"),
        ("fl_lora", "local_lora"),
        ("fedsa_lora", "local_lora"),
        ("fixed_share_b_lora", "local_lora"),
        ("fixed_share_b_lora", "fedsa_lora"),
    )
    output = []
    for left_key, right_key in pairs:
        left, right = by_key[left_key], by_key[right_key]
        output.append(
            {
                "comparison": f"{left['method']} - {right['method']}",
                "left_method_key": left_key,
                "right_method_key": right_key,
                **{
                    f"delta_{metric}": float(left[metric]) - float(right[metric])
                    for metric in ("AP", "AP50", "AP75")
                },
            }
        )
    return output


def _csv_text(rows: Sequence[Mapping], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _fmt(value) -> str:
    return "---" if value is None else f"{float(value):.4f}"


def _markdown(case: Mapping, rows: Sequence[Mapping], comparisons: Sequence[Mapping]) -> str:
    lines = [
        "# Client-local zero-positive-support class case",
        "",
        (
            "This is a descriptive naturally occurring partition case "
            "(`n=1`), not a repeated leave-one-class-out experiment."
        ),
        "",
        "## Case definition",
        "",
        f"- Seed/client/class: `{case['seed']}/{case['client_id']}/{case['class_name']}`",
        f"- Local train positive boxes/images: `{case['train_boxes']}/{case['train_images']}`",
        f"- Local validation positive boxes/images: `{case['val_boxes']}/{case['val_images']}`",
        (
            "- Local test positive boxes/images: "
            f"`{case['local_test_boxes']}/{case['local_test_images']}`"
        ),
        f"- Other-client train positive boxes: `{case['other_client_train_boxes']}`",
        (
            "- Other-client train positive boxes by client: `"
            f"{case['other_client_train_boxes_by_client']}`"
        ),
        f"- Common pooled-test GT boxes: `{rows[0]['common_test_support']}`",
        "",
        "## Common pooled-test performance",
        "",
        (
            "| Method | Model scope | Other-client H boxes contributing signal | "
            "Information path | AP | AP50 | AP75 | Delta AP vs matched Local |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {model_scope} | "
            "{other_client_train_boxes_contributing_signal} | "
            "{information_path} | {AP:.4f} | {AP50:.4f} | {AP75:.4f} | {delta} |".format(
                **row, delta=_fmt(row["delta_AP_vs_matched_local"])
            )
        )
    lines.extend(
        [
            "",
            "## Descriptive paired differences",
            "",
            "| Comparison | Delta AP | Delta AP50 | Delta AP75 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {row['delta_AP']:+.4f} | "
            f"{row['delta_AP50']:+.4f} | {row['delta_AP75']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `Unseen` means zero positive annotation in the focal client's training split; it is not open-vocabulary or general zero-shot detection.",
            "- Centralized models directly use pooled raw training data and are contextual references, not locally-unseen training methods.",
            "- Federated differences can indicate cross-client knowledge transfer, but every FL method also communicates the task head; effects cannot be attributed to A or B alone.",
            "- For FL rows, other-client boxes contribute only through aggregated parameter updates; raw boxes and images are not transmitted.",
            "- Validation support is disclosed because checkpoint selection consults validation AP even though validation annotations do not produce gradient updates.",
            "- This single seed-client-class observation has no run SD, confidence interval, or significance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run(results_root: Path, split_dir: Path, output_dir: Path) -> dict:
    manifests, split_hashes = load_split_manifests(split_dir)
    cases = discover_zero_support_cases(manifests)
    signatures = [
        (case["seed"], case["client_id"], case["class_name"]) for case in cases
    ]
    if signatures != [EXPECTED_CASE]:
        raise ValueError(
            "Expected exactly the data-only primary zero-support case "
            f"{EXPECTED_CASE}, found {signatures}"
        )
    records, input_hashes = collect_primary_results(
        results_root, split_dir, manifests, split_hashes
    )
    case = cases[0]
    rows = extract_case_rows(results_root, records, case)
    comparisons = build_comparisons(rows)

    # All validation and extraction complete before creating publication files.
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "client_local_zero_positive_support_common_test_case",
        "case_n": 1,
        "descriptive_only": True,
        "primary_files_modified": False,
        "case": case,
        "rows": rows,
        "comparisons": comparisons,
        "interpretation_boundary": [
            "not open-vocabulary or general zero-shot detection",
            "no mean, run SD, confidence interval, or significance claim",
            "task head is shared in every federated protocol",
            "centralized rows are pooled-data contextual references",
        ],
    }
    row_fields = (
        "method_key",
        "method",
        "model_scope",
        "information_path",
        "seed",
        "client_id",
        "class_name",
        "local_train_boxes",
        "local_train_images",
        "local_val_boxes",
        "local_val_images",
        "local_test_boxes",
        "local_test_images",
        "other_client_train_boxes",
        "other_client_train_boxes_contributing_signal",
        "common_test_support",
        "AP",
        "AP50",
        "AP75",
        "baseline_key",
        "delta_AP_vs_matched_local",
        "result_file",
        "result_file_sha256",
    )
    comparison_fields = (
        "comparison",
        "left_method_key",
        "right_method_key",
        "delta_AP",
        "delta_AP50",
        "delta_AP75",
    )

    # Recheck every protected input before publishing any derived output.
    current_hashes = {path: _sha256(path) for path in input_hashes}
    changed = [
        str(path)
        for path in input_hashes
        if current_hashes[path] != input_hashes[path]
    ]
    if changed:
        raise RuntimeError(
            f"Primary inputs changed during read-only analysis: {changed}"
        )

    outputs = {
        "unseen_class_case.csv": _csv_text(rows, row_fields),
        "comparisons.csv": _csv_text(comparisons, comparison_fields),
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        "summary.md": _markdown(case, rows, comparisons),
    }
    for name, text in outputs.items():
        _atomic_write(output_dir / name, text)

    current_hashes = {path: _sha256(path) for path in input_hashes}
    changed = [
        str(path)
        for path in input_hashes
        if current_hashes[path] != input_hashes[path]
    ]
    if changed:
        raise RuntimeError(f"Primary inputs changed during read-only analysis: {changed}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "input_file_count": len(input_hashes),
        "input_files_byte_identical": True,
        "input_sha256": {
            str(path.resolve()): digest
            for path, digest in sorted(input_hashes.items(), key=lambda item: str(item[0]))
        },
        "output_sha256": {
            name: _sha256(output_dir / name) for name in sorted(outputs)
        },
    }
    _atomic_write(
        output_dir / "summary_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.results_root.resolve(),
        args.split_dir.resolve(),
        args.output_dir.resolve(),
    )
    case = summary["case"]
    print(
        "[PASS] unique zero-local-positive-support case: "
        f"seed={case['seed']} client={case['client_id']} class={case['class_name']}"
    )
    for row in summary["rows"]:
        print(
            f"{row['method']:20s} "
            f"AP/AP50/AP75={row['AP']:.4f}/{row['AP50']:.4f}/{row['AP75']:.4f} "
            f"delta_vs_local={_fmt(row['delta_AP_vs_matched_local'])}"
        )
    print(f"[PASS] Primary JSON/split files are byte-identical")
    print(f"[Results] {args.output_dir.resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
