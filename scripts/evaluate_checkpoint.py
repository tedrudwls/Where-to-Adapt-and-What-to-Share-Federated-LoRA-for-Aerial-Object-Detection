#!/usr/bin/env python3
"""Read-only evaluation of the frozen representative federated checkpoint.

This P0 entry point intentionally supports one fully specified vertical slice:
``seed_42/fl_fedsa_lora_r8_a0.4``.  It verifies every model/protocol artifact
before deserializing PyTorch state, rebuilds a test-only YOLO view under a
temporary directory, restores all three personalized endpoints, and evaluates
both own-client and pooled common-test AP.  It never calls ``prepare_data()`` or
``evaluate_federated_checkpoint()`` because those historical paths may create
caches or rewrite a primary result JSON.

Only the final machine-readable report is written to stdout.  Runtime/model
messages go to stderr.  No persistent output option is provided by design.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
import random
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "artifacts" / "checkpoint_index.json"
DEFAULT_REFERENCE = (
    PROJECT_ROOT / "artifacts" / "evaluation_reference_seed42_fedsa_lora_r8_a04.json"
)
TARGET_EXPERIMENT_ID = "seed_42/fl_fedsa_lora_r8_a0.4"
EVALUATION_SCHEMA_VERSION = 1
REPORT_METRICS = ("AP", "AP50", "AP75")
SHA256_LENGTH = 64
CHECKPOINT_SCHEMA = 5
CHECKPOINT_KIND = "federated_personalized"
FEDSA_PAYLOAD_POLICY = "global_A_plus_global_task_head__local_B"
FROZEN_HISTORICAL_CHECKPOINT_BYTES = 3316516
FROZEN_HISTORICAL_CHECKPOINT_SHA256 = (
    "3ed025419506009465add698da75fef941c20c0b16eb347bb224820781b9cac4"
)
FROZEN_PUBLIC_CHECKPOINT_BYTES = 3316452
FROZEN_PUBLIC_CHECKPOINT_SHA256 = (
    "391205473ad8de24af56ba1b566e54a6f305dd0b79806cb468583e84e464fa14"
)
FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256 = (
    "b42e1e811238caf1ec76e788547dbcf46d8f390b3b0e63864cf3d303e88c6d4f"
)
FROZEN_PUBLIC_TENSOR_COUNT = 231
FROZEN_PRETRAINED_SHA256 = (
    "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
)
FROZEN_SPLIT_SHA256 = (
    "74a45a37f4b05c474564993ff15f5548875f3e767abc19a0d2d30f195e1754c5"
)
FROZEN_ARCHIVED_RESULT_BYTES = 5297013
FROZEN_ARCHIVED_RESULT_SHA256 = (
    "ed2b53790bc3a44a157ce61f078332723be388744af54ec315cac3badbfa2c4d"
)
FROZEN_REFERENCE_SHA256 = (
    "4f6d33eeb255a96d0f49c51600dcf546a2ac5d96e782bd9826094f848e98dbef"
)
FROZEN_AUGMENTATION_PROTOCOL = {
    "implementation": "ultralytics_8.4.126_RTDETRDataset",
    "initial": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
    },
    "close_mosaic_effective_epochs_requested": 10,
    "close_mosaic_disables": ["mosaic", "mixup", "cutmix", "copy_paste"],
    "rect": False,
    "cache": False,
    "train_only": True,
    "persistent_dataloader_workers": False,
}
FROZEN_COMPATIBILITY = {
    "fl_method": "fedsa_lora",
    "model_name": "rtdetr-l",
    "num_classes": 4,
    "num_clients": 3,
    "fl_rounds": 20,
    "local_epochs": 5,
    "batch_size": 8,
    "img_size": 640,
    "num_workers": 4,
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
    "augmentation_protocol": FROZEN_AUGMENTATION_PROTOCOL,
    "seed": 42,
    "partition_seed": 42,
    "partition": "dirichlet",
    "dirichlet_alpha": 0.4,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.0,
    "apply_lora_backbone": True,
    "apply_lora_decoder": True,
    "backbone_min_channels": 64,
}
FROZEN_CHECKPOINT_CONTRACT = {
    "schema_version": CHECKPOINT_SCHEMA,
    "checkpoint_kind": CHECKPOINT_KIND,
    "resume_capability": "evaluation_only_no_optimizer_scheduler_or_rng_state",
    "selection": "best_macro_client_local_val_AP",
    "training_rounds_executed": 20,
    "best_round": 20,
    "nonfloating_state_policy": "retain_previous_server_value",
    "federated_payload_policy": FEDSA_PAYLOAD_POLICY,
    "shared_lora_factor_role": "A",
    "local_lora_factor_role": "B",
}
FROZEN_MANIFEST_PROTOCOL = {
    "schema_version": 7,
    "partition": "dirichlet",
    "dirichlet_alpha": 0.4,
    "partition_algorithm": "source_group_target_deficit_balance_cross_split_owner_v2",
    "client_partition_unit": "source_group",
    "num_clients": 3,
    "seed": 42,
    "min_bbox_area": 0.0,
    "min_bbox_side": 0.0,
    "drop_empty_images": False,
    "crowd_policy": "require_zero_crowd_annotations_for_YOLO_metric_equivalence",
    "category_policy": "exact_aod4_targets_ignore_only_unreferenced_declared_categories",
    "source_split_policy": "official_aod4_v6",
    "official_split_preserved": True,
    "source_identity_policy": "roboflow_source_key_or_exact_sha256_connected_components",
    "source_split_priority": [],
    "image_hash_check_enabled": True,
    "class_names": ["airplane", "bird", "drone", "helicopter"],
    "cat_id_to_label": {"1": 0, "2": 1, "3": 2, "4": 3},
}


class EvaluationError(RuntimeError):
    """A deterministic validation or evaluation-contract failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, label: str) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationError(f"{label} must be a JSON object: {path}")
    return payload


def _require_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise EvaluationError(f"Missing {label}: {path}") from error
    if not resolved.is_file():
        raise EvaluationError(f"{label} is not a regular file: {path}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise EvaluationError(f"Missing {label}: {path}") from error
    if not resolved.is_dir():
        raise EvaluationError(f"{label} is not a directory: {path}")
    return resolved


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_bytes: Optional[int] = None,
) -> dict:
    path = _require_file(path, label)
    if not _is_sha256(expected_sha256):
        raise EvaluationError(f"Invalid expected SHA-256 for {label}")
    actual_bytes = int(path.stat().st_size)
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise EvaluationError(
            f"{label} byte-size mismatch: expected={expected_bytes}, actual={actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise EvaluationError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )
    return {
        "file_name": path.name,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_bytes: Optional[int] = None,
) -> Path:
    """Copy one opened input into private storage and verify the copied bytes."""

    source = _require_file(source, label)
    if not _is_sha256(expected_sha256):
        raise EvaluationError(f"Invalid expected SHA-256 for {label}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for chunk in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as error:
        raise EvaluationError(f"Cannot stage verified {label}: {error}") from error
    actual_sha256 = digest.hexdigest()
    if expected_bytes is not None and byte_count != int(expected_bytes):
        raise EvaluationError(
            f"Staged {label} byte-size mismatch: expected={expected_bytes}, "
            f"actual={byte_count}"
        )
    if actual_sha256 != expected_sha256:
        raise EvaluationError(
            f"Staged {label} SHA-256 mismatch: expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )
    destination.chmod(stat.S_IRUSR)
    return destination


def _load_checkpoint_index(path: Path) -> tuple[dict, list]:
    index = _load_json(path, "checkpoint index")
    records = index.get("records")
    if index.get("schema_version") != 1 or not isinstance(records, list) or not records:
        raise EvaluationError("Checkpoint index must use schema_version=1 with records")
    if int(index.get("record_count", -1)) != len(records):
        raise EvaluationError("Checkpoint index record_count mismatch")
    total_bytes = 0
    seen_experiments, seen_assets, seen_paths, seen_digests = set(), set(), set(), set()
    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise EvaluationError(f"Checkpoint index record {number} is not an object")
        experiment_id = record.get("experiment_id")
        asset_name = record.get("release_asset_name")
        historical_path = record.get("historical_project_relative_path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise EvaluationError(f"Checkpoint index record {number} has no experiment_id")
        if (
            not isinstance(asset_name, str)
            or not asset_name
            or Path(asset_name).name != asset_name
        ):
            raise EvaluationError(f"Checkpoint index record {number} has invalid asset name")
        if not isinstance(historical_path, str) or not historical_path:
            raise EvaluationError(f"Checkpoint index record {number} has no historical path")
        pure_path = PurePosixPath(historical_path)
        if (
            pure_path.is_absolute()
            or "\\" in historical_path
            or "//" in historical_path
            or any(part in ("", ".", "..") for part in pure_path.parts)
        ):
            raise EvaluationError(f"Checkpoint index record {number} has unsafe path")
        if not _is_sha256(digest):
            raise EvaluationError(f"Checkpoint index record {number} has invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise EvaluationError(f"Checkpoint index record {number} has invalid byte size")
        if (
            experiment_id in seen_experiments
            or asset_name in seen_assets
            or historical_path in seen_paths
            or digest in seen_digests
        ):
            raise EvaluationError(
                "Checkpoint index duplicates an experiment, asset, path, or SHA-256"
            )
        seen_experiments.add(experiment_id)
        seen_assets.add(asset_name)
        seen_paths.add(historical_path)
        seen_digests.add(digest)
        total_bytes += size
    if int(index.get("total_bytes", -1)) != total_bytes:
        raise EvaluationError("Checkpoint index total_bytes mismatch")
    return index, records


def _select_target_record(records: Sequence[dict], experiment_id: str) -> dict:
    matches = [record for record in records if record.get("experiment_id") == experiment_id]
    if len(matches) != 1:
        raise EvaluationError(
            f"Expected one checkpoint-index record for {experiment_id}, found {len(matches)}"
        )
    record = matches[0]
    expected = {
        "mode": "fl",
        "method": "fedsa_lora",
        "partition": "dirichlet",
        "partition_seed": 42,
        "training_seed": 42,
        "rank": 8,
        "selected_unit": "round",
        "selected_at": 20,
        "release_asset_name": (
            "seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt"
        ),
        "historical_project_relative_path": (
            "results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/"
            "weights/best_federated.pt"
        ),
        "bytes": FROZEN_HISTORICAL_CHECKPOINT_BYTES,
        "sha256": FROZEN_HISTORICAL_CHECKPOINT_SHA256,
        "pretrained_sha256": FROZEN_PRETRAINED_SHA256,
        "split_manifest_sha256": FROZEN_SPLIT_SHA256,
    }
    mismatches = {
        key: {"index": record.get(key), "required": value}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise EvaluationError(
            "Representative checkpoint index contract changed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not _is_sha256(record.get("pretrained_sha256")):
        raise EvaluationError("Target index record has no valid pretrained SHA-256")
    if not _is_sha256(record.get("split_manifest_sha256")):
        raise EvaluationError("Target index record has no valid split-manifest SHA-256")
    return record


def _validate_reference(reference: dict, record: dict) -> dict:
    if reference.get("schema_version") != 1:
        raise EvaluationError("Evaluation reference must use schema_version=1")
    if reference.get("protocol") != "read_only_representative_checkpoint_evaluation":
        raise EvaluationError("Evaluation reference protocol is not the frozen protocol")
    if reference.get("experiment_id") != TARGET_EXPERIMENT_ID:
        raise EvaluationError("Evaluation reference targets a different experiment")
    if reference.get("method") != "fedsa_lora":
        raise EvaluationError("Evaluation reference method must be fedsa_lora")
    if reference.get("metric_scale") != "0_to_1":
        raise EvaluationError("Evaluation reference metric scale must be 0_to_1")
    historical_checkpoint = reference.get("historical_checkpoint")
    public_checkpoint = reference.get("public_checkpoint")
    pretrained = reference.get("pretrained_model")
    split = reference.get("split_manifest")
    archived = reference.get("archived_result")
    public_replay = reference.get("public_replay_manifest")
    expected = reference.get("expected")
    if not all(isinstance(value, dict) for value in (
        historical_checkpoint, public_checkpoint, pretrained, split, archived,
        public_replay, expected
    )):
        raise EvaluationError("Evaluation reference is incomplete")
    if (
        public_replay.get("file_name")
        != "seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json"
        or not _is_sha256(public_replay.get("sha256"))
        or isinstance(public_replay.get("bytes"), bool)
        or not isinstance(public_replay.get("bytes"), int)
        or public_replay["bytes"] <= 0
    ):
        raise EvaluationError("Evaluation reference public replay identity is invalid")
    public_historical = public_checkpoint.get("historical_source")
    public_tensor = public_checkpoint.get("tensor_fingerprint")
    if not isinstance(public_historical, dict) or not isinstance(public_tensor, dict):
        raise EvaluationError(
            "Evaluation reference public-checkpoint provenance is incomplete"
        )
    cross_checks = {
        "release_asset_name": (
            historical_checkpoint.get("release_asset_name"),
            record.get("release_asset_name"),
        ),
        "historical_project_relative_path": (
            historical_checkpoint.get("historical_project_relative_path"),
            record.get("historical_project_relative_path"),
        ),
        "checkpoint.bytes": (
            historical_checkpoint.get("bytes"), record.get("bytes")
        ),
        "checkpoint.sha256": (
            historical_checkpoint.get("sha256"), record.get("sha256")
        ),
        "checkpoint.selected_round": (
            historical_checkpoint.get("selected_round"), record.get("selected_at")
        ),
        "pretrained.sha256": (
            pretrained.get("sha256"), record.get("pretrained_sha256")
        ),
        "split.sha256": (
            split.get("sha256"), record.get("split_manifest_sha256")
        ),
    }
    mismatches = {
        key: {"reference": left, "index": right}
        for key, (left, right) in cross_checks.items()
        if left != right
    }
    if mismatches:
        raise EvaluationError(
            "Evaluation reference and checkpoint index disagree: "
            + json.dumps(mismatches, sort_keys=True)
        )
    public_checks = {
        "file_name": (
            public_checkpoint.get("file_name"), record.get("release_asset_name")
        ),
        "bytes": (
            public_checkpoint.get("bytes"), FROZEN_PUBLIC_CHECKPOINT_BYTES
        ),
        "sha256": (
            public_checkpoint.get("sha256"), FROZEN_PUBLIC_CHECKPOINT_SHA256
        ),
        "historical.bytes": (
            public_historical.get("bytes"),
            FROZEN_HISTORICAL_CHECKPOINT_BYTES,
        ),
        "historical.sha256": (
            public_historical.get("sha256"),
            FROZEN_HISTORICAL_CHECKPOINT_SHA256,
        ),
        "tensor.algorithm": (
            public_tensor.get("algorithm"),
            "recursive_path_dtype_shape_raw_bytes_sha256_v1",
        ),
        "tensor.sha256": (
            public_tensor.get("sha256"),
            FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256,
        ),
        "tensor.count": (
            public_tensor.get("tensor_count"),
            FROZEN_PUBLIC_TENSOR_COUNT,
        ),
    }
    public_mismatches = {
        key: {"reference": left, "required": right}
        for key, (left, right) in public_checks.items()
        if left != right
    }
    if public_mismatches:
        raise EvaluationError(
            "Evaluation reference public-checkpoint identity changed: "
            + json.dumps(public_mismatches, sort_keys=True)
        )
    checkpoint_contract = historical_checkpoint.get("contract")
    if checkpoint_contract != FROZEN_CHECKPOINT_CONTRACT:
        raise EvaluationError(
            "Evaluation reference checkpoint contract differs from the frozen target"
        )
    compatibility = historical_checkpoint.get("compatibility")
    if compatibility != FROZEN_COMPATIBILITY:
        raise EvaluationError(
            "Evaluation reference compatibility differs from the frozen target"
        )
    manifest_protocol = split.get("protocol")
    if manifest_protocol != FROZEN_MANIFEST_PROTOCOL:
        raise EvaluationError(
            "Evaluation reference split protocol differs from the frozen target"
        )
    split_counts = split.get("split_counts")
    client_image_counts = split.get("client_image_counts")
    source_inventory = split.get("source_inventory")
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "val", "test"}:
        raise EvaluationError("Evaluation reference split counts are incomplete")
    if (
        not isinstance(client_image_counts, dict)
        or set(client_image_counts) != {"train", "val", "test"}
    ):
        raise EvaluationError("Evaluation reference client image counts are incomplete")
    if (
        not isinstance(source_inventory, dict)
        or source_inventory.get("identity_policy")
        != FROZEN_MANIFEST_PROTOCOL["source_identity_policy"]
        or not _is_sha256(source_inventory.get("inventory_sha256"))
        or not _is_sha256(source_inventory.get("test_image_tree_sha256"))
    ):
        raise EvaluationError("Evaluation reference source inventory is incomplete")
    for split_name in ("train", "val", "test"):
        counts = split_counts[split_name]
        per_client = client_image_counts[split_name]
        if (
            not isinstance(counts, dict)
            or isinstance(counts.get("images"), bool)
            or not isinstance(counts.get("images"), int)
            or counts["images"] <= 0
            or isinstance(counts.get("annotations"), bool)
            or not isinstance(counts.get("annotations"), int)
            or counts["annotations"] < 0
        ):
            raise EvaluationError(
                f"Evaluation reference has invalid {split_name} split counts"
            )
        if (
            not isinstance(per_client, list)
            or len(per_client) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in per_client)
            or sum(per_client) != counts["images"]
        ):
            raise EvaluationError(
                f"Evaluation reference has invalid {split_name} client image counts"
            )
    if not _is_sha256(archived.get("sha256")):
        raise EvaluationError("Evaluation reference has invalid result SHA-256")
    if not isinstance(archived.get("bytes"), int) or archived["bytes"] <= 0:
        raise EvaluationError("Evaluation reference has invalid result byte size")
    if (
        archived.get("sha256") != FROZEN_ARCHIVED_RESULT_SHA256
        or archived.get("bytes") != FROZEN_ARCHIVED_RESULT_BYTES
    ):
        raise EvaluationError("Evaluation reference archived-result identity changed")
    tolerance = reference.get("absolute_tolerance")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        raise EvaluationError("Evaluation reference has invalid absolute_tolerance")
    if tolerance <= 0 or tolerance > 1e-4:
        raise EvaluationError("Evaluation tolerance must be in (0, 1e-4]")
    _validate_normalized_metrics(expected, label="reference expected metrics")
    return reference


def _metric_triplet(record: Mapping[str, Any]) -> dict:
    values = {}
    for metric in REPORT_METRICS:
        try:
            value = float(record[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"Missing/non-numeric metric {metric}") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EvaluationError(f"Metric {metric} is outside [0,1]: {value}")
        values[metric] = value
    return values


def _normalize_archived_result(payload: dict, checkpoint_reference: dict) -> dict:
    if payload.get("status") != "complete":
        raise EvaluationError("Archived result is not complete")
    if payload.get("mode") != "fl" or payload.get("fl_method") != "fedsa_lora":
        raise EvaluationError("Archived result is not the representative FL FedSA run")
    if int(payload.get("seed", -1)) != 42 or int(payload.get("partition_seed", -1)) != 42:
        raise EvaluationError("Archived result seed tuple is not (42,42)")
    archived_checks = {
        "partition": (payload.get("partition"), "dirichlet"),
        "dirichlet_alpha": (payload.get("dirichlet_alpha"), 0.4),
        "num_clients": (payload.get("num_clients"), 3),
        "rounds_planned": (payload.get("rounds_planned"), 20),
        "rounds_executed": (payload.get("rounds_executed"), 20),
        "local_epochs": (payload.get("local_epochs"), 5),
        "federated_checkpoint_schema_version": (
            payload.get("federated_checkpoint_schema_version"), CHECKPOINT_SCHEMA
        ),
        "federated_payload_policy": (
            payload.get("federated_payload_policy"), FEDSA_PAYLOAD_POLICY
        ),
        "shared_lora_factor_role": (payload.get("shared_lora_factor_role"), "A"),
        "client_local_lora_factor_role": (
            payload.get("client_local_lora_factor_role"), "B"
        ),
    }
    archived_mismatches = {
        key: {"archived_result": left, "required": right}
        for key, (left, right) in archived_checks.items()
        if left != right
    }
    if archived_mismatches:
        raise EvaluationError(
            "Archived result protocol mismatch: "
            + json.dumps(archived_mismatches, sort_keys=True)
        )
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise EvaluationError("Archived result has no checkpoint selection block")
    selection_checks = {
        "criterion": (
            selection.get("criterion"), FROZEN_CHECKPOINT_CONTRACT["selection"]
        ),
        "round": (selection.get("round"), checkpoint_reference.get("selected_round")),
        "checkpoint.basename": (
            os.path.basename(str(selection.get("checkpoint", ""))),
            os.path.basename(
                str(checkpoint_reference.get("historical_project_relative_path", ""))
            ),
        ),
    }
    selection_mismatches = {
        key: {"archived_result": left, "required": right}
        for key, (left, right) in selection_checks.items()
        if left != right
    }
    if selection_mismatches:
        raise EvaluationError(
            "Archived result checkpoint selection mismatch: "
            + json.dumps(selection_mismatches, sort_keys=True)
        )
    local = payload.get("client_local_test")
    common = payload.get("common_test")
    summary = payload.get("client_summary")
    if not isinstance(local, list) or len(local) != 3:
        raise EvaluationError("Archived result must contain three client-local records")
    if not isinstance(common, dict) or not isinstance(summary, dict):
        raise EvaluationError("Archived result lacks common/client summary records")
    common_rows = common.get("per_client_model")
    if not isinstance(common_rows, list) or len(common_rows) != 3:
        raise EvaluationError("Archived result must contain three common-test records")
    local_output = []
    common_output = []
    for expected_id, row in enumerate(local):
        if int(row.get("client_id", -1)) != expected_id:
            raise EvaluationError("Archived client-local records are out of order")
        local_output.append({
            "client_id": expected_id,
            "num_images": int(row["num_images"]),
            **_metric_triplet(row),
        })
    for expected_id, row in enumerate(common_rows):
        if int(row.get("client_id", -1)) != expected_id:
            raise EvaluationError("Archived common-test records are out of order")
        common_output.append({"client_id": expected_id, **_metric_triplet(row)})
    try:
        client_macro = {
            metric: float(summary[metric]["macro_mean"])
            for metric in REPORT_METRICS
        }
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError("Archived client macro metrics are incomplete") from error
    return {
        "client_local_test": local_output,
        "client_macro": _metric_triplet(client_macro),
        "common_per_client_model": common_output,
        "common_macro": _metric_triplet(common),
    }


def _validate_normalized_metrics(metrics: dict, *, label: str) -> None:
    if not isinstance(metrics, dict):
        raise EvaluationError(f"{label} is not an object")
    local = metrics.get("client_local_test")
    common = metrics.get("common_per_client_model")
    if not isinstance(local, list) or len(local) != 3:
        raise EvaluationError(f"{label} must contain three client-local rows")
    if not isinstance(common, list) or len(common) != 3:
        raise EvaluationError(f"{label} must contain three common-test rows")
    for rows, include_count in ((local, True), (common, False)):
        for client_id, row in enumerate(rows):
            if not isinstance(row, dict) or int(row.get("client_id", -1)) != client_id:
                raise EvaluationError(f"{label} client rows are incomplete or out of order")
            if include_count and int(row.get("num_images", 0)) <= 0:
                raise EvaluationError(f"{label} has an invalid client image count")
            _metric_triplet(row)
    _metric_triplet(metrics.get("client_macro", {}))
    _metric_triplet(metrics.get("common_macro", {}))


def _assert_metrics_equal(left: dict, right: dict, *, tolerance: float, label: str) -> None:
    differences = _metric_differences(left, right)
    maximum = max((item["absolute_error"] for item in differences), default=0.0)
    if maximum > tolerance:
        worst = max(differences, key=lambda item: item["absolute_error"])
        raise EvaluationError(
            f"{label} differs from the frozen reference: max_abs_error={maximum:.9g}, "
            f"path={worst['path']}, tolerance={tolerance:.9g}"
        )


def _metric_differences(actual: dict, expected: dict) -> list:
    differences = []
    for section in ("client_local_test", "common_per_client_model"):
        for actual_row, expected_row in zip(actual[section], expected[section]):
            client_id = int(expected_row["client_id"])
            if int(actual_row["client_id"]) != client_id:
                raise EvaluationError(f"Actual {section} client order changed")
            if section == "client_local_test" and (
                int(actual_row["num_images"]) != int(expected_row["num_images"])
            ):
                raise EvaluationError(f"Actual client {client_id} image count changed")
            for metric in REPORT_METRICS:
                actual_value = float(actual_row[metric])
                expected_value = float(expected_row[metric])
                differences.append({
                    "path": f"{section}[{client_id}].{metric}",
                    "actual": actual_value,
                    "expected": expected_value,
                    "absolute_error": abs(actual_value - expected_value),
                })
    for section in ("client_macro", "common_macro"):
        for metric in REPORT_METRICS:
            actual_value = float(actual[section][metric])
            expected_value = float(expected[section][metric])
            differences.append({
                "path": f"{section}.{metric}",
                "actual": actual_value,
                "expected": expected_value,
                "absolute_error": abs(actual_value - expected_value),
            })
    return differences


def _safe_relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluationError(f"Unsafe dataset filename: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise EvaluationError(f"Unsafe dataset filename: {value!r}")
    return pure.as_posix()


def _inventory_tree_digest(records: Sequence[list]) -> str:
    digest = hashlib.sha256()
    normalized = sorted(records, key=lambda row: (str(row[1]), int(row[0])))
    for _, relative_name, content_digest in normalized:
        digest.update(str(relative_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_digest).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple:
    value = path.stat()
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        stat.S_IMODE(value.st_mode),
    )


def _require_exact_keys(value: Any, keys: set, label: str) -> dict:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != keys:
        raise EvaluationError(
            f"{label} keys differ from the public replay schema: "
            f"missing={sorted(keys - observed)}, extra={sorted(observed - keys)}"
        )
    return value


def _public_replay_as_validation_manifest(
    replay: dict, manifest_reference: dict
) -> dict:
    """Validate the compact public schema and adapt it to the legacy validator.

    Train/validation image IDs are intentionally not public because inference
    only needs their frozen per-client counts.  The legacy validator receives
    count-only blocks for those splits and still validates every test ID/hash.
    """

    replay = _require_exact_keys(
        replay,
        {
            "schema_version", "protocol", "experiment_id",
            "historical_split_manifest", "partition_protocol", "split_counts",
            "client_image_counts", "test",
        },
        "public replay manifest",
    )
    if replay["schema_version"] != 1:
        raise EvaluationError("Public replay manifest must use schema_version=1")
    if replay["protocol"] != "fedlora_representative_test_replay":
        raise EvaluationError("Public replay manifest protocol changed")
    if replay["experiment_id"] != TARGET_EXPERIMENT_ID:
        raise EvaluationError("Public replay manifest targets another experiment")
    historical = _require_exact_keys(
        replay["historical_split_manifest"], {"schema_version", "sha256"},
        "public replay historical split",
    )
    if (
        historical["schema_version"] != FROZEN_MANIFEST_PROTOCOL["schema_version"]
        or historical["sha256"] != FROZEN_SPLIT_SHA256
    ):
        raise EvaluationError("Public replay historical split identity changed")
    if replay["partition_protocol"] != manifest_reference.get("protocol"):
        raise EvaluationError("Public replay partition protocol changed")
    if replay["split_counts"] != manifest_reference.get("split_counts"):
        raise EvaluationError("Public replay aggregate split counts changed")
    if replay["client_image_counts"] != manifest_reference.get("client_image_counts"):
        raise EvaluationError("Public replay client image counts changed")

    test = _require_exact_keys(
        replay["test"],
        {"annotation", "category_audit", "source_inventory", "client_image_ids"},
        "public replay test block",
    )
    annotation = _require_exact_keys(
        test["annotation"], {"relative_path", "sha256"},
        "public replay test annotation",
    )
    if annotation["relative_path"] != "test/_annotations.coco.json":
        raise EvaluationError("Public replay annotation path changed")
    if not _is_sha256(annotation["sha256"]):
        raise EvaluationError("Public replay annotation SHA-256 is invalid")
    inventory = _require_exact_keys(
        test["source_inventory"],
        {
            "schema_version", "identity_policy", "historical_inventory_sha256",
            "image_tree_sha256", "records",
        },
        "public replay test inventory",
    )
    reference_inventory = manifest_reference.get("source_inventory", {})
    inventory_checks = {
        "schema_version": (inventory["schema_version"], 1),
        "identity_policy": (
            inventory["identity_policy"], reference_inventory.get("identity_policy")
        ),
        "historical_inventory_sha256": (
            inventory["historical_inventory_sha256"],
            reference_inventory.get("inventory_sha256"),
        ),
        "image_tree_sha256": (
            inventory["image_tree_sha256"],
            reference_inventory.get("test_image_tree_sha256"),
        ),
    }
    mismatches = {
        key: {"replay": left, "required": right}
        for key, (left, right) in inventory_checks.items() if left != right
    }
    if mismatches:
        raise EvaluationError(
            "Public replay source inventory changed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    client_rows = test["client_image_ids"]
    if not isinstance(client_rows, list) or len(client_rows) != 3:
        raise EvaluationError("Public replay must contain three client test assignments")
    ordered_rows = sorted(client_rows, key=lambda row: int(row.get("client_id", -1)))
    if [int(row.get("client_id", -1)) for row in ordered_rows] != [0, 1, 2]:
        raise EvaluationError("Public replay client test IDs must be 0,1,2")

    counts = replay["client_image_counts"]
    clients = []
    for client_id, row in enumerate(ordered_rows):
        row = _require_exact_keys(
            row, {"client_id", "image_ids"},
            f"public replay client {client_id} test assignment",
        )
        image_ids = row["image_ids"]
        if not isinstance(image_ids, list):
            raise EvaluationError("Public replay client image_ids must be a list")
        clients.append({
            "client_id": client_id,
            "splits": {
                "train": {"num_images": int(counts["train"][client_id])},
                "val": {"num_images": int(counts["val"][client_id])},
                "test": {
                    "num_images": int(counts["test"][client_id]),
                    "image_ids": image_ids,
                },
            },
        })
    realized = {
        split_name: [
            {"client_id": client_id, "num_images": int(counts[split_name][client_id])}
            for client_id in range(3)
        ]
        for split_name in ("train", "val", "test")
    }
    metadata = {
        **copy.deepcopy(replay["partition_protocol"]),
        "data_root": "",
        "annotation_sha256": {"test": annotation["sha256"]},
        "split_counts": copy.deepcopy(replay["split_counts"]),
        "source_split_counts": copy.deepcopy(replay["split_counts"]),
        "realized_partition_statistics": realized,
        "source_category_audit": {"test": copy.deepcopy(test["category_audit"])},
        "source_hash_inventory": {
            "schema_version": 1,
            "identity_policy": inventory["identity_policy"],
            "inventory_sha256": inventory["historical_inventory_sha256"],
            "records": {"test": copy.deepcopy(inventory["records"])},
            "per_split_image_tree_sha256": {
                "test": inventory["image_tree_sha256"]
            },
        },
    }
    return {"metadata": metadata, "clients": clients}


def _validate_relocated_test_data(
    manifest: dict,
    data_root: Path,
    manifest_reference: dict,
    *,
    counts_only_non_test: bool = False,
) -> dict:
    metadata = manifest.get("metadata")
    clients = manifest.get("clients")
    if not isinstance(metadata, dict):
        raise EvaluationError("Split manifest has no metadata object")
    expected_protocol = manifest_reference.get("protocol")
    expected_split_counts = manifest_reference.get("split_counts")
    expected_client_counts = manifest_reference.get("client_image_counts")
    if not all(isinstance(value, dict) for value in (
        expected_protocol, expected_split_counts, expected_client_counts
    )):
        raise EvaluationError("Evaluation reference has no complete split protocol")
    protocol_mismatches = {
        key: {"manifest": metadata.get(key), "required": value}
        for key, value in expected_protocol.items()
        if metadata.get(key) != value
    }
    if protocol_mismatches:
        raise EvaluationError(
            "Split manifest protocol mismatch: "
            + json.dumps(protocol_mismatches, sort_keys=True)
        )
    if metadata.get("split_counts") != expected_split_counts:
        raise EvaluationError("Split manifest aggregate counts differ from the frozen reference")
    if metadata.get("source_split_counts") != expected_split_counts:
        raise EvaluationError(
            "Split manifest source counts differ from the frozen official-split reference"
        )
    if not isinstance(clients, list) or len(clients) != 3:
        raise EvaluationError("Split manifest must describe exactly three clients")
    class_names = metadata.get("class_names")
    if class_names != FROZEN_MANIFEST_PROTOCOL["class_names"]:
        raise EvaluationError("Split manifest class names differ from AOD-4")
    test_root = data_root / "test"
    annotation = _require_file(test_root / "_annotations.coco.json", "test annotation")
    expected_annotation_sha = metadata.get("annotation_sha256", {}).get("test")
    annotation_record = _verify_file(
        annotation,
        label="test annotation",
        expected_sha256=expected_annotation_sha,
    )
    coco = _load_json(annotation, "test COCO annotation")
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise EvaluationError(f"Test COCO annotation has no list-valued {key}")
    declared = {
        str(int(category["id"])): str(category["name"])
        for category in coco["categories"]
    }
    stored_audit = metadata.get("source_category_audit", {}).get("test", {})
    if declared != stored_audit.get("declared_categories"):
        raise EvaluationError("Relocated test COCO categories differ from the manifest")
    image_by_id = {}
    for image in coco["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise EvaluationError(f"Duplicate test COCO image id: {image_id}")
        image_by_id[image_id] = image
    expected_counts = expected_split_counts["test"]
    if len(image_by_id) != int(expected_counts.get("images", -1)):
        raise EvaluationError("Relocated test COCO image count differs from the manifest")
    if len(coco["annotations"]) != int(expected_counts.get("annotations", -1)):
        raise EvaluationError("Relocated test COCO annotation count differs from the manifest")
    category_map = {
        int(key): int(value)
        for key, value in metadata.get("cat_id_to_label", {}).items()
    }
    raw_counts = {str(key): 0 for key in declared}
    for annotation_row in coco["annotations"]:
        image_id = int(annotation_row["image_id"])
        category_id = int(annotation_row["category_id"])
        if image_id not in image_by_id:
            raise EvaluationError(f"Annotation references unknown test image {image_id}")
        if category_id not in category_map:
            raise EvaluationError(f"Annotation uses unmapped category {category_id}")
        if int(bool(annotation_row.get("iscrowd", 0))) != 0:
            raise EvaluationError("The frozen AOD-4 test protocol contains no crowd boxes")
        raw_counts[str(category_id)] = raw_counts.get(str(category_id), 0) + 1
    if raw_counts != stored_audit.get("raw_annotation_counts"):
        raise EvaluationError("Relocated test class counts differ from the manifest")

    inventory = metadata.get("source_hash_inventory")
    if not isinstance(inventory, dict) or int(inventory.get("schema_version", -1)) != 1:
        raise EvaluationError("Split manifest has no schema-1 source inventory")
    records = inventory.get("records", {}).get("test")
    expected_tree_digest = inventory.get("per_split_image_tree_sha256", {}).get("test")
    reference_inventory = manifest_reference["source_inventory"]
    if (
        inventory.get("identity_policy") != reference_inventory["identity_policy"]
        or inventory.get("inventory_sha256") != reference_inventory["inventory_sha256"]
        or expected_tree_digest != reference_inventory["test_image_tree_sha256"]
    ):
        raise EvaluationError("Split manifest source-inventory identity changed")
    if not isinstance(records, list) or not _is_sha256(expected_tree_digest):
        raise EvaluationError("Split manifest has no valid test image inventory")
    if _inventory_tree_digest(records) != expected_tree_digest:
        raise EvaluationError("Manifest test image inventory digest is internally inconsistent")
    inventory_by_id = {}
    image_paths = {}
    image_stats = {}
    for row in records:
        if not isinstance(row, list) or len(row) != 3:
            raise EvaluationError("Malformed test image inventory row")
        image_id, relative_name, digest = int(row[0]), _safe_relative_name(row[1]), row[2]
        if image_id in inventory_by_id or not _is_sha256(digest):
            raise EvaluationError("Duplicate image id or invalid SHA in test inventory")
        inventory_by_id[image_id] = (relative_name, digest)
    if set(inventory_by_id) != set(image_by_id):
        raise EvaluationError("Test image inventory IDs differ from COCO")
    for image_id, image in image_by_id.items():
        relative_name = _safe_relative_name(image.get("file_name"))
        expected_name, expected_digest = inventory_by_id[image_id]
        if relative_name != expected_name:
            raise EvaluationError(f"Test image filename mismatch for image {image_id}")
        candidate = test_root.joinpath(*PurePosixPath(relative_name).parts)
        candidate = _require_file(candidate, f"test image {image_id}")
        try:
            candidate.relative_to(test_root.resolve())
        except ValueError as error:
            raise EvaluationError(f"Test image resolves outside data root: {relative_name}") from error
        actual_digest = sha256_file(candidate)
        if actual_digest != expected_digest:
            raise EvaluationError(
                f"Test image SHA-256 mismatch: image_id={image_id}, file={relative_name}"
            )
        image_paths[image_id] = candidate
        image_stats[relative_name] = _stat_signature(candidate)

    ordered_clients = sorted(clients, key=lambda item: int(item["client_id"]))
    if [int(item["client_id"]) for item in ordered_clients] != [0, 1, 2]:
        raise EvaluationError("Split manifest client IDs must be 0,1,2")
    realized = metadata.get("realized_partition_statistics")
    if not isinstance(realized, dict) or set(realized) != {"train", "val", "test"}:
        raise EvaluationError(
            "Split manifest realized partition statistics are incomplete"
        )
    assignments = {split_name: [] for split_name in ("train", "val", "test")}
    client_test_ids, client_train_sizes, client_eval_sizes = [], [], []
    for client_id, client in enumerate(ordered_clients):
        split_blocks = client.get("splits", {})
        if not isinstance(split_blocks, dict):
            raise EvaluationError(f"Client {client_id} has no split blocks")
        for split_name in ("train", "val", "test"):
            block = split_blocks.get(split_name)
            if not isinstance(block, dict):
                raise EvaluationError(
                    f"Client {client_id} has no {split_name} split block"
                )
            expected_client_count = expected_client_counts[split_name][client_id]
            if counts_only_non_test and split_name != "test":
                if set(block) != {"num_images"}:
                    raise EvaluationError(
                        f"Public replay client {client_id} {split_name} block "
                        "must contain only num_images"
                    )
                if int(block.get("num_images", -1)) != int(expected_client_count):
                    raise EvaluationError(
                        f"Client {client_id} {split_name} count mismatch"
                    )
                image_ids = []
            else:
                image_ids = [int(value) for value in block.get("image_ids", [])]
                if len(image_ids) != len(set(image_ids)):
                    raise EvaluationError(
                        f"Client {client_id} {split_name} assignment has duplicates"
                    )
                if (
                    len(image_ids) != int(block.get("num_images", -1))
                    or len(image_ids) != int(expected_client_count)
                ):
                    raise EvaluationError(
                        f"Client {client_id} {split_name} assignment count mismatch"
                    )
            assignments[split_name].append(image_ids)
            realized_rows = realized.get(split_name)
            if not isinstance(realized_rows, list) or len(realized_rows) != 3:
                raise EvaluationError(
                    f"Split manifest {split_name} realized statistics are incomplete"
                )
            realized_row = realized_rows[client_id]
            if (
                not isinstance(realized_row, dict)
                or int(realized_row.get("client_id", -1)) != client_id
                or int(realized_row.get("num_images", -1)) != expected_client_count
            ):
                raise EvaluationError(
                    f"Client {client_id} {split_name} realized count mismatch"
                )
        test_ids = assignments["test"][client_id]
        client_test_ids.append(test_ids)
        client_train_sizes.append(int(expected_client_counts["train"][client_id]))
        client_eval_sizes.append({
            "val": int(expected_client_counts["val"][client_id]),
            "test": int(expected_client_counts["test"][client_id]),
        })
    for split_name, per_client in assignments.items():
        if counts_only_non_test and split_name != "test":
            continue
        flattened = [image_id for values in per_client for image_id in values]
        if (
            len(flattened) != len(set(flattened))
            or len(flattened) != int(expected_split_counts[split_name]["images"])
        ):
            raise EvaluationError(
                f"Client {split_name} assignments are not a disjoint complete count"
            )
    flattened = [image_id for values in client_test_ids for image_id in values]
    if set(flattened) != set(image_by_id):
        raise EvaluationError("Client test assignments are not a disjoint complete cover")
    return {
        "coco": coco,
        "annotation_path": annotation,
        "annotation_record": annotation_record,
        "test_root": test_root.resolve(),
        "class_names": list(class_names),
        "category_map": category_map,
        "image_paths": image_paths,
        "image_stats": image_stats,
        "image_tree_sha256": expected_tree_digest,
        "client_test_ids": client_test_ids,
        "client_train_sizes": client_train_sizes,
        "client_eval_sizes": client_eval_sizes,
        "historical_data_root": str(metadata.get("data_root", "")),
    }


def _clip_bbox(bbox: Sequence[Any], width: int, height: int) -> Optional[tuple]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise EvaluationError(f"Malformed COCO bbox: {bbox!r}")
    x, y, box_width, box_height = map(float, bbox)
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        raise EvaluationError(f"Non-finite COCO bbox: {bbox!r}")
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _write_yolo_subset(
    *,
    coco: dict,
    image_paths: Mapping[int, Path],
    image_ids: Iterable[int],
    category_map: Mapping[int, int],
    output_dir: Path,
    class_names: Sequence[str],
) -> Path:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations_by_image: Dict[int, list] = {image_id: [] for image_id in image_by_id}
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    for image_id in image_ids:
        image = image_by_id[int(image_id)]
        relative_name = _safe_relative_name(image["file_name"])
        relative_path = PurePosixPath(relative_name)
        destination = images_dir.joinpath(*relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(image_paths[int(image_id)]), str(destination))
        label_relative = relative_path.with_suffix(".txt")
        label_path = labels_dir.joinpath(*label_relative.parts)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            raise EvaluationError(f"Invalid dimensions for test image {image_id}")
        for annotation in annotations_by_image[int(image_id)]:
            clipped = _clip_bbox(annotation["bbox"], width, height)
            if clipped is None:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in category_map:
                raise EvaluationError(f"Unmapped category id {category_id}")
            cx, cy, norm_width, norm_height = clipped
            rows.append(
                f"{category_map[category_id]} {cx:.8f} {cy:.8f} "
                f"{norm_width:.8f} {norm_height:.8f}\n"
            )
        label_path.write_text("".join(rows), encoding="utf-8")
    yaml_path = output_dir / "dataset.yaml"
    names = "\n".join(
        f"  {index}: {json.dumps(name, ensure_ascii=False)}"
        for index, name in enumerate(class_names)
    )
    yaml_path.write_text(
        "\n".join((
            f"path: {json.dumps(str(output_dir.resolve()))}",
            "train: images",
            "val: images",
            "test: images",
            f"nc: {len(class_names)}",
            "names:",
            names,
            "",
        )),
        encoding="utf-8",
    )
    return yaml_path


def _build_temporary_data_info(
    data_spec: dict,
    manifest: dict,
    split_sha256: str,
    runtime_root: Path,
) -> dict:
    client_yamls = []
    for client_id, image_ids in enumerate(data_spec["client_test_ids"]):
        yaml_path = _write_yolo_subset(
            coco=data_spec["coco"],
            image_paths=data_spec["image_paths"],
            image_ids=image_ids,
            category_map=data_spec["category_map"],
            output_dir=runtime_root / f"client_{client_id}",
            class_names=data_spec["class_names"],
        )
        client_yamls.append(str(yaml_path))
    full_yaml = _write_yolo_subset(
        coco=data_spec["coco"],
        image_paths=data_spec["image_paths"],
        image_ids=sorted(data_spec["image_paths"]),
        category_map=data_spec["category_map"],
        output_dir=runtime_root / "full",
        class_names=data_spec["class_names"],
    )
    return {
        "client_yamls": client_yamls,
        "client_sizes": list(data_spec["client_train_sizes"]),
        "client_eval_sizes": list(data_spec["client_eval_sizes"]),
        "full_yaml": str(full_yaml),
        "class_names": list(data_spec["class_names"]),
        "split_manifest_sha256": split_sha256,
        "split_metadata": manifest["metadata"],
    }


def _snapshot_protected(paths: Sequence[Path]) -> dict:
    snapshot = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        value = resolved.stat()
        snapshot[str(resolved)] = {
            "bytes": int(value.st_size),
            "mtime_ns": int(value.st_mtime_ns),
            "ctime_ns": int(value.st_ctime_ns),
            "mode": stat.S_IMODE(value.st_mode),
            "sha256": sha256_file(resolved),
        }
    return snapshot


def _assert_protected_unchanged(before: dict) -> None:
    changes = []
    for raw_path, expected in before.items():
        path = Path(raw_path)
        if not path.is_file():
            changes.append({"path": raw_path, "status": "missing"})
            continue
        value = path.stat()
        actual = {
            "bytes": int(value.st_size),
            "mtime_ns": int(value.st_mtime_ns),
            "ctime_ns": int(value.st_ctime_ns),
            "mode": stat.S_IMODE(value.st_mode),
            "sha256": sha256_file(path),
        }
        if actual != expected:
            changes.append({"path": raw_path, "status": "changed"})
    if changes:
        raise EvaluationError(
            "Protected input changed during evaluation: "
            + json.dumps(changes, sort_keys=True)
        )


def _assert_test_images_unchanged(data_spec: dict) -> None:
    changes = []
    for relative_name, expected in data_spec["image_stats"].items():
        path = data_spec["test_root"].joinpath(*PurePosixPath(relative_name).parts)
        if not path.is_file() or _stat_signature(path) != expected:
            changes.append(relative_name)
            if len(changes) == 10:
                break
    if changes:
        raise EvaluationError(
            f"Relocated test image metadata changed during evaluation: {changes}"
        )


def _tree_metadata_snapshot(root: Path) -> dict:
    snapshot = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = {"kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            snapshot[relative] = {"kind": "file", "stat": _stat_signature(path)}
        elif path.is_dir():
            snapshot[relative] = {"kind": "directory", "stat": _stat_signature(path)}
        else:
            snapshot[relative] = {"kind": "other", "stat": _stat_signature(path)}
    return snapshot


def _assert_tree_metadata_unchanged(root: Path, before: dict) -> None:
    after = _tree_metadata_snapshot(root)
    if after == before:
        return
    added = sorted(set(after) - set(before))[:10]
    removed = sorted(set(before) - set(after))[:10]
    changed = sorted(
        key for key in set(before) & set(after) if before[key] != after[key]
    )[:10]
    raise EvaluationError(
        "Relocated test tree changed during evaluation: "
        + json.dumps(
            {"added": added, "removed": removed, "changed": changed},
            sort_keys=True,
        )
    )


@contextlib.contextmanager
def _isolated_runtime(root: Path):
    environment = {
        "YOLO_CONFIG_DIR": root / "ultralytics_config",
        "XDG_CACHE_HOME": root / "xdg_cache",
        "MPLCONFIGDIR": root / "matplotlib_config",
        "TORCH_HOME": root / "torch_home",
        "TRITON_CACHE_DIR": root / "triton_cache",
        "CUDA_CACHE_PATH": root / "cuda_cache",
    }
    previous = {key: os.environ.get(key) for key in environment}
    old_cwd = Path.cwd()
    try:
        for key, path in environment.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        os.chdir(root)
        yield
    finally:
        os.chdir(old_cwd)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _redirect_all_stdout_to_stderr():
    """Keep stdout JSON-only, including native writes to file descriptor 1."""

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, OSError):
        pass
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (AttributeError, OSError):
            pass
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def _args_from_compatibility(
    compatibility: dict,
    *,
    data_root: Path,
    split_file: Path,
    model_weights: Path,
    device: str,
) -> SimpleNamespace:
    required = {
        "fl_method", "model_name", "num_classes", "num_clients", "fl_rounds",
        "local_epochs", "batch_size", "img_size", "num_workers", "lr", "head_lr",
        "backbone_lr_ratio", "weight_decay", "warmup_epochs", "min_lr_ratio",
        "grad_clip_norm", "close_mosaic_epochs", "fedprox_mu",
        "reset_optimizer_each_round", "amp", "seed", "partition_seed",
        "partition", "dirichlet_alpha", "lora_rank", "lora_alpha", "lora_dropout",
        "apply_lora_backbone", "apply_lora_decoder", "backbone_min_channels",
    }
    missing = sorted(required - set(compatibility))
    if missing:
        raise EvaluationError(f"Checkpoint compatibility manifest is incomplete: {missing}")
    values = dict(compatibility)
    values.update({
        "data_root": str(data_root),
        "split_file": str(split_file),
        "model_weights": str(model_weights),
        "device": device,
        "min_bbox_area": 0.0,
        "min_bbox_side": 0.0,
        "rehash_source_images": False,
        "run_mia": False,
        "cross_client_eval": False,
        "visualize_interval": 0,
        "vis_samples": 0,
        "mia_max_samples": 1000,
        "mia_calibration_fraction": 0.5,
        "patience": 0,
        "val_interval": 5,
    })
    return SimpleNamespace(**values)


def _validate_checkpoint_payload(payload: Any, record: dict, data_info: dict) -> dict:
    if not isinstance(payload, dict):
        raise EvaluationError("Checkpoint payload is not a dictionary")
    checks = {
        "schema_version": (payload.get("schema_version"), CHECKPOINT_SCHEMA),
        "checkpoint_kind": (payload.get("checkpoint_kind"), CHECKPOINT_KIND),
        "fl_method": (payload.get("fl_method"), "fedsa_lora"),
        "num_clients": (payload.get("num_clients"), 3),
        "round": (payload.get("round"), int(record["selected_at"])),
        "split_manifest_sha256": (
            payload.get("split_manifest_sha256"), data_info["split_manifest_sha256"]
        ),
        "federated_payload_policy": (
            payload.get("federated_payload_policy"), FEDSA_PAYLOAD_POLICY
        ),
        "shared_lora_factor_role": (payload.get("shared_lora_factor_role"), "A"),
        "local_lora_factor_role": (payload.get("local_lora_factor_role"), "B"),
        "class_names": (payload.get("class_names"), data_info["class_names"]),
        "client_sample_counts": (
            payload.get("client_sample_counts"), data_info["client_sizes"]
        ),
    }
    mismatches = {
        key: {"checkpoint": left, "required": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    if mismatches:
        raise EvaluationError(
            "Representative checkpoint payload contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not isinstance(payload.get("shared_state"), dict) or not payload["shared_state"]:
        raise EvaluationError("Checkpoint has no shared tensor state")
    local_states = payload.get("local_personalized_states")
    if not isinstance(local_states, list) or len(local_states) != 3:
        raise EvaluationError("Checkpoint must contain one local-B state per client")
    if any(not isinstance(state, dict) or not state for state in local_states):
        raise EvaluationError("Checkpoint contains an invalid client-local B state")
    contract_mismatches = {
        key: {"checkpoint": payload.get(key), "required": value}
        for key, value in FROZEN_CHECKPOINT_CONTRACT.items()
        if payload.get(key) != value
    }
    if contract_mismatches:
        raise EvaluationError(
            "Representative checkpoint selection/contract mismatch: "
            + json.dumps(contract_mismatches, sort_keys=True)
        )
    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, dict):
        raise EvaluationError("Checkpoint has no strict compatibility manifest")
    if compatibility != FROZEN_COMPATIBILITY:
        differing_keys = sorted(
            key
            for key in set(compatibility) | set(FROZEN_COMPATIBILITY)
            if compatibility.get(key) != FROZEN_COMPATIBILITY.get(key)
        )
        raise EvaluationError(
            "Representative checkpoint compatibility mismatch: "
            + json.dumps(differing_keys)
        )
    return payload


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _perform_model_evaluation(
    *,
    checkpoint: Path,
    record: dict,
    model_weights: Path,
    split_file: Path,
    data_root: Path,
    data_info: dict,
    device: str,
) -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import torch
        from trainers.fl_server import (
            _apply_checkpoint,
            _evaluate_client_local,
            _evaluate_common_test,
            _new_client_models,
        )
    except ImportError as error:
        raise EvaluationError(
            "Evaluation dependencies are unavailable; install requirements.txt"
        ) from error
    try:
        payload = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise EvaluationError(
            f"Restricted checkpoint deserialization failed: {error}"
        ) from error
    payload = _validate_checkpoint_payload(payload, record, data_info)
    args = _args_from_compatibility(
        payload["compatibility"],
        data_root=data_root,
        split_file=split_file,
        model_weights=model_weights,
        device=device,
    )
    _set_seed(args.seed)
    client_models = _new_client_models(args, data_info)
    _apply_checkpoint(payload, client_models, args, data_info)
    local_rows, local_summary = _evaluate_client_local(
        client_models, data_info, args, split="test"
    )
    common = _evaluate_common_test(client_models, data_info, args)
    normalized = {
        "client_local_test": [
            {
                "client_id": int(row["client_id"]),
                "num_images": int(row["num_images"]),
                **_metric_triplet(row),
            }
            for row in local_rows
        ],
        "client_macro": {
            metric: float(local_summary[metric]["macro_mean"])
            for metric in REPORT_METRICS
        },
        "common_per_client_model": [
            {"client_id": int(row["client_id"]), **_metric_triplet(row)}
            for row in common["per_client_model"]
        ],
        "common_macro": _metric_triplet(common),
    }
    _validate_normalized_metrics(normalized, label="recomputed metrics")
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--reference-result",
        type=Path,
        help=(
            "Optional immutable historical result JSON for author-side provenance "
            "verification. Public replay uses the code-pinned compact reference."
        ),
    )
    parser.add_argument("--model-weights", type=Path, required=True)
    manifests = parser.add_mutually_exclusive_group(required=True)
    manifests.add_argument(
        "--split-file", type=Path,
        help="Original schema-v7 split manifest (legacy author-side audit mode)",
    )
    manifests.add_argument(
        "--replay-manifest", type=Path,
        help="Path-free public replay manifest distributed with the checkpoint",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experiment-id",
        default=TARGET_EXPERIMENT_ID,
        choices=(TARGET_EXPERIMENT_ID,),
    )
    return parser


def evaluate(argv: Optional[List[str]] = None, *, evaluator=None) -> tuple[int, Optional[dict]]:
    args = _build_parser().parse_args(argv)
    checkpoint = _require_file(args.checkpoint, "checkpoint")
    index_path = _require_file(args.checkpoint_index, "checkpoint index")
    reference_path = _require_file(args.reference, "evaluation reference")
    reference_result_path = (
        _require_file(args.reference_result, "archived result")
        if args.reference_result is not None else None
    )
    model_weights = _require_file(args.model_weights, "pretrained model")
    historical_mode = args.split_file is not None
    if historical_mode and reference_result_path is None:
        raise EvaluationError(
            "--reference-result is required with the historical --split-file mode"
        )
    manifest_path = _require_file(
        args.split_file if historical_mode else args.replay_manifest,
        "split manifest" if historical_mode else "public replay manifest",
    )
    data_root = _require_directory(args.data_root, "data root")
    test_root = _require_directory(data_root / "test", "relocated test split")
    annotation_path = _require_file(
        test_root / "_annotations.coco.json", "test annotation"
    )
    protected_paths = [
        checkpoint, index_path, reference_path, model_weights, manifest_path,
        annotation_path,
    ]
    if reference_result_path is not None:
        protected_paths.append(reference_result_path)
    # Capture the baseline before parsing, hashing, or deserializing any input.
    # This closes the verification-to-snapshot gap and makes failures auditable.
    protected_before = _snapshot_protected(protected_paths)
    test_tree_before = _tree_metadata_snapshot(test_root)

    chosen_evaluator = evaluator or _perform_model_evaluation
    data_spec = None
    result = None
    try:
        index, records = _load_checkpoint_index(index_path)
        record = _select_target_record(records, args.experiment_id)
        reference_file_record = _verify_file(
            reference_path,
            label="evaluation reference",
            expected_sha256=FROZEN_REFERENCE_SHA256,
        )
        reference = _validate_reference(
            _load_json(reference_path, "evaluation reference"), record
        )
        if historical_mode:
            checkpoint_variant = "historical"
            checkpoint_sha256 = FROZEN_HISTORICAL_CHECKPOINT_SHA256
            checkpoint_bytes = FROZEN_HISTORICAL_CHECKPOINT_BYTES
        else:
            checkpoint_variant = "public_sanitized"
            checkpoint_sha256 = FROZEN_PUBLIC_CHECKPOINT_SHA256
            checkpoint_bytes = FROZEN_PUBLIC_CHECKPOINT_BYTES
        checkpoint_record = _verify_file(
            checkpoint,
            label=f"{checkpoint_variant} checkpoint",
            expected_sha256=checkpoint_sha256,
            expected_bytes=checkpoint_bytes,
        )
        model_record = _verify_file(
            model_weights,
            label="pretrained model",
            expected_sha256=FROZEN_PRETRAINED_SHA256,
        )
        tolerance = float(reference["absolute_tolerance"])
        archived_record = None
        if reference_result_path is not None:
            archived_record = _verify_file(
                reference_result_path,
                label="archived result",
                expected_sha256=FROZEN_ARCHIVED_RESULT_SHA256,
                expected_bytes=FROZEN_ARCHIVED_RESULT_BYTES,
            )
            archived_metrics = _normalize_archived_result(
                _load_json(reference_result_path, "archived result"),
                reference["historical_checkpoint"],
            )
            _assert_metrics_equal(
                archived_metrics,
                reference["expected"],
                tolerance=1e-12,
                label="Archived result",
            )
        if historical_mode:
            manifest_record = _verify_file(
                manifest_path,
                label="split manifest",
                expected_sha256=FROZEN_SPLIT_SHA256,
            )
            manifest = _load_json(manifest_path, "split manifest")
            counts_only_non_test = False
        else:
            replay_identity = reference["public_replay_manifest"]
            manifest_record = _verify_file(
                manifest_path,
                label="public replay manifest",
                expected_sha256=replay_identity["sha256"],
                expected_bytes=replay_identity["bytes"],
            )
            manifest = _public_replay_as_validation_manifest(
                _load_json(manifest_path, "public replay manifest"),
                reference["split_manifest"],
            )
            counts_only_non_test = True
        data_spec = _validate_relocated_test_data(
            manifest,
            data_root,
            reference["split_manifest"],
            counts_only_non_test=counts_only_non_test,
        )

        # Refuse to begin model construction if any input changed while the
        # validation pass was running. The finally block repeats these gates.
        _assert_protected_unchanged(protected_before)
        _assert_test_images_unchanged(data_spec)
        _assert_tree_metadata_unchanged(test_root, test_tree_before)

        with tempfile.TemporaryDirectory(prefix="fedlora-readonly-eval-") as directory:
            runtime_root = Path(directory)
            trusted_checkpoint = _copy_verified_file(
                checkpoint,
                runtime_root / "trusted_inputs" / "best_federated.pt",
                label=f"{checkpoint_variant} checkpoint",
                expected_sha256=checkpoint_sha256,
                expected_bytes=checkpoint_bytes,
            )
            trusted_model_weights = _copy_verified_file(
                model_weights,
                runtime_root / "trusted_inputs" / "rtdetr-l.pt",
                label="pretrained model",
                expected_sha256=FROZEN_PRETRAINED_SHA256,
            )
            _assert_protected_unchanged(protected_before)
            with _isolated_runtime(runtime_root):
                runtime_data = _build_temporary_data_info(
                    data_spec,
                    manifest,
                    FROZEN_SPLIT_SHA256,
                    runtime_root / "test_only_data",
                )
                with _redirect_all_stdout_to_stderr():
                    actual_metrics = chosen_evaluator(
                        checkpoint=trusted_checkpoint,
                        record=record,
                        model_weights=trusted_model_weights,
                        split_file=manifest_path,
                        data_root=data_root,
                        data_info=runtime_data,
                        device=args.device,
                    )
                _validate_normalized_metrics(actual_metrics, label="recomputed metrics")
                differences = _metric_differences(actual_metrics, reference["expected"])
                max_abs_error = max(item["absolute_error"] for item in differences)
                comparison_status = "pass" if max_abs_error <= tolerance else "fail"
                result = {
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "status": comparison_status,
                    "read_only": True,
                    "experiment_id": args.experiment_id,
                    "method": reference["method"],
                    "display_name": reference["display_name"],
                    "metric_scale": "0_to_1",
                    "runtime": {
                        "device": args.device,
                        "temporary_test_only_data": True,
                        "private_verified_model_copies": True,
                        "persistent_outputs": False,
                        "historical_data_root": (
                            data_spec["historical_data_root"] if historical_mode else None
                        ),
                        "relocated_data_root_used": (
                            str(data_root) if historical_mode else None
                        ),
                        "manifest_mode": (
                            "historical_schema_v7" if historical_mode else "public_replay"
                        ),
                    },
                    "inputs": {
                        "checkpoint": {
                            **checkpoint_record,
                            "variant": checkpoint_variant,
                        },
                        "checkpoint_index": {
                            "file_name": index_path.name,
                            "sha256": sha256_file(index_path),
                            "record_count": int(index["record_count"]),
                        },
                        "evaluation_reference": reference_file_record,
                        "pretrained_model": model_record,
                        "historical_split_manifest_sha256": FROZEN_SPLIT_SHA256,
                        (
                            "split_manifest" if historical_mode
                            else "public_replay_manifest"
                        ): manifest_record,
                        "archived_result": archived_record,
                        "archived_result_verified": archived_record is not None,
                        "test_annotation": data_spec["annotation_record"],
                        "test_image_tree_sha256": data_spec["image_tree_sha256"],
                        "test_images_verified": len(data_spec["image_paths"]),
                    },
                    "metrics": actual_metrics,
                    "reference_comparison": {
                        "absolute_tolerance": tolerance,
                        "max_absolute_error": max_abs_error,
                        "passed": comparison_status == "pass",
                        "differences": differences,
                    },
                }
    finally:
        _assert_protected_unchanged(protected_before)
        if data_spec is not None:
            _assert_test_images_unchanged(data_spec)
        _assert_tree_metadata_unchanged(test_root, test_tree_before)
    if result is None:
        raise EvaluationError("Evaluation produced no report")
    result["integrity_gate"] = {
        "status": "pass",
        "protected_files_unchanged": len(protected_before),
        "test_image_metadata_unchanged": len(data_spec["image_stats"]),
    }
    return (0 if result["status"] == "pass" else 1), result


def main(argv: Optional[List[str]] = None) -> int:
    try:
        code, report = evaluate(argv)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    if report is None:
        print("[FAIL] Evaluation returned no report", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
