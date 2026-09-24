#!/usr/bin/env python3
"""Read-only pre-publication audit for the 12 paper-core checkpoints.

The audit selects exactly four primary FL methods at seeds 42/43/44 from the
committed checkpoint index.  Each historical checkpoint is verified before a
restricted ``torch.load(..., weights_only=True)``, checked against its method-
specific checkpoint contract, and sanitized in memory.  The resulting report
records the deterministic candidate identity of every sanitized checkpoint;
it does not write a checkpoint or build a release archive.

Historical checkpoints and the checkpoint index are hashed again after all
deserialization and in-memory serialization.  The JSON report is emitted only
when every protected source remains byte-identical, and it never contains an
absolute local filesystem path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_representative_release_bundle as release_builder  # noqa: E402
from scripts.verify_checkpoint_assets import load_index  # noqa: E402


AUDIT_SCHEMA_VERSION = 1
AUDIT_POLICY = "paper_core_12_checkpoint_read_only_prepublication_audit"
SEEDS = (42, 43, 44)
METHOD_ORDER = ("full_ft", "lora", "fedsa_lora", "fixed_share_b_lora")
PAPER_METHOD_NAMES = {
    "full_ft": "FL Full FT",
    "lora": "FedLoRA-AB",
    "fedsa_lora": "FedLoRA-A",
    "fixed_share_b_lora": "FedLoRA-B",
}
DEFAULT_SPEC = PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json"
PRETRAINED_SHA256 = (
    "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
)
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

METHOD_CONTRACTS = {
    "full_ft": {
        "experiment_suffix": "fl_full_ft_a0.4",
        "rank": None,
        "payload_policy": "global_full_model_state",
        "shared_role": None,
        "local_role": None,
    },
    "lora": {
        "experiment_suffix": "fl_lora_r8_a0.4",
        "rank": 8,
        "payload_policy": "global_A_B_plus_global_task_head",
        "shared_role": "A+B",
        "local_role": None,
    },
    "fedsa_lora": {
        "experiment_suffix": "fl_fedsa_lora_r8_a0.4",
        "rank": 8,
        "payload_policy": "global_A_plus_global_task_head__local_B",
        "shared_role": "A",
        "local_role": "B",
    },
    "fixed_share_b_lora": {
        "experiment_suffix": "fl_fixed_share_b_lora_r8_a0.4",
        "rank": 8,
        "payload_policy": "global_B_plus_global_task_head__local_A",
        "shared_role": "B",
        "local_role": "A",
    },
}


class CoreAuditError(RuntimeError):
    """A fail-closed paper-core input audit error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _expected_experiment_id(seed: int, method: str) -> str:
    return f"seed_{seed}/{METHOD_CONTRACTS[method]['experiment_suffix']}"


def _safe_relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        raise CoreAuditError(f"{label} is not a canonical relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise CoreAuditError(f"{label} is not a safe relative path")
    return pure.as_posix()


def select_core_records(records: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Select and semantically validate the exact 4-method x 3-seed grid."""

    expected = {
        _expected_experiment_id(seed, method): (seed, method)
        for seed in SEEDS
        for method in METHOD_ORDER
    }
    by_experiment: dict[str, Mapping[str, Any]] = {}
    for record in records:
        experiment_id = record.get("experiment_id")
        if experiment_id in expected:
            if experiment_id in by_experiment:
                raise CoreAuditError(f"Duplicate core record: {experiment_id}")
            by_experiment[str(experiment_id)] = record
    missing = sorted(set(expected) - set(by_experiment))
    if missing:
        raise CoreAuditError(f"Checkpoint index is missing core records: {missing}")

    selected = []
    for seed in SEEDS:
        for method in METHOD_ORDER:
            experiment_id = _expected_experiment_id(seed, method)
            record = dict(by_experiment[experiment_id])
            contract = METHOD_CONTRACTS[method]
            checks = {
                "mode": (record.get("mode"), "fl"),
                "method": (record.get("method"), method),
                "partition": (record.get("partition"), "dirichlet"),
                "training_seed": (record.get("training_seed"), seed),
                "partition_seed": (record.get("partition_seed"), seed),
                "rank": (record.get("rank"), contract["rank"]),
                "client_id": (record.get("client_id"), None),
                "selected_unit": (record.get("selected_unit"), "round"),
                "pretrained_sha256": (
                    record.get("pretrained_sha256"), PRETRAINED_SHA256
                ),
            }
            mismatches = {
                key: {"index": left, "required": right}
                for key, (left, right) in checks.items()
                if left != right
            }
            if mismatches:
                raise CoreAuditError(
                    f"Core index contract mismatch for {experiment_id}: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            selected_at = record.get("selected_at")
            if (
                isinstance(selected_at, bool)
                or not isinstance(selected_at, int)
                or not 1 <= selected_at <= 20
            ):
                raise CoreAuditError(f"Invalid selected round for {experiment_id}")
            if not SHA256_PATTERN.fullmatch(str(record.get("split_manifest_sha256", ""))):
                raise CoreAuditError(f"Invalid split SHA-256 for {experiment_id}")
            _safe_relative_path(
                record.get("historical_project_relative_path"),
                f"{experiment_id} historical path",
            )
            asset = record.get("release_asset_name")
            if not isinstance(asset, str) or Path(asset).name != asset:
                raise CoreAuditError(f"Invalid release asset name for {experiment_id}")
            selected.append(record)
    if len(selected) != 12:
        raise CoreAuditError("Paper-core selection did not produce exactly 12 records")
    return selected


def _load_release_spec(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoreAuditError(f"Cannot read paper-core release spec: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CoreAuditError("Paper-core release spec must use schema_version=1")
    if value.get("status") not in {
        "public_identity_discovery_required",
        "public_identities_pinned",
    }:
        raise CoreAuditError("Paper-core release spec has an unsupported status")
    if value.get("expected_record_count") != 12:
        raise CoreAuditError("Paper-core release spec must require 12 records")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise CoreAuditError("Paper-core release spec must contain 12 records")
    if any(not isinstance(row, dict) for row in records):
        raise CoreAuditError("Paper-core release spec contains a malformed record")
    return value


def _validate_spec_against_index(
    spec: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], index_identity: Mapping[str, Any]
) -> None:
    index_block = spec.get("checkpoint_index")
    if not isinstance(index_block, dict):
        raise CoreAuditError("Paper-core release spec has no checkpoint-index identity")
    expected_index_path = _safe_relative_path(
        index_block.get("path"), "release spec checkpoint-index path"
    )
    if expected_index_path != index_identity["relative_path"]:
        raise CoreAuditError("Paper-core release spec points to a different checkpoint index")
    if index_block.get("sha256") != index_identity["sha256"]:
        raise CoreAuditError("Checkpoint-index SHA-256 differs from the release spec")
    if spec.get("historical_total_bytes") != sum(int(row["bytes"]) for row in selected):
        raise CoreAuditError("Release-spec historical byte total is inconsistent")
    pretrained = spec.get("external_pretrained_model")
    if (
        not isinstance(pretrained, dict)
        or pretrained.get("included") is not False
        or pretrained.get("file_name") != "rtdetr-l.pt"
        or pretrained.get("sha256") != PRETRAINED_SHA256
    ):
        raise CoreAuditError("Release-spec pretrained-model contract changed")

    by_experiment = {row["experiment_id"]: row for row in selected}
    spec_records = spec["records"]
    if [row.get("experiment_id") for row in spec_records] != [
        row["experiment_id"] for row in selected
    ]:
        raise CoreAuditError("Release-spec record order/scope differs from the index")
    for row in spec_records:
        experiment_id = row.get("experiment_id")
        indexed = by_experiment.get(experiment_id)
        historical = row.get("historical")
        if not isinstance(historical, dict):
            raise CoreAuditError(f"Release spec has no historical block: {experiment_id}")
        checks = {
            "paper_method": (
                row.get("paper_method"), PAPER_METHOD_NAMES[indexed["method"]]
            ),
            "internal_method": (row.get("internal_method"), indexed["method"]),
            "training_seed": (row.get("training_seed"), indexed["training_seed"]),
            "partition_seed": (row.get("partition_seed"), indexed["partition_seed"]),
            "rank": (row.get("rank"), indexed["rank"]),
            "selected_round": (row.get("selected_round"), indexed["selected_at"]),
            "split_manifest_sha256": (
                row.get("split_manifest_sha256"), indexed["split_manifest_sha256"]
            ),
            "historical.project_relative_path": (
                historical.get("project_relative_path"),
                indexed["historical_project_relative_path"],
            ),
            "historical.bytes": (historical.get("bytes"), indexed["bytes"]),
            "historical.sha256": (historical.get("sha256"), indexed["sha256"]),
            "public_file_name": (
                row.get("public_file_name"), indexed["release_asset_name"]
            ),
        }
        mismatches = {
            key: {"spec": left, "index": right}
            for key, (left, right) in checks.items()
            if left != right
        }
        if mismatches:
            raise CoreAuditError(
                f"Release-spec/index mismatch for {experiment_id}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        public_identity = row.get("public_identity")
        if spec["status"] == "public_identity_discovery_required":
            if public_identity is not None:
                raise CoreAuditError(
                    f"Discovery spec unexpectedly pins a public identity: {experiment_id}"
                )
        else:
            if not isinstance(public_identity, dict):
                raise CoreAuditError(
                    f"Pinned spec has no public identity: {experiment_id}"
                )
            tensor = public_identity.get("tensor_fingerprint")
            if (
                set(public_identity) != {
                    "bytes", "sha256", "tensor_fingerprint",
                    "path_replacements", "residual_absolute_path_count",
                }
                or isinstance(public_identity.get("bytes"), bool)
                or not isinstance(public_identity.get("bytes"), int)
                or public_identity["bytes"] <= 0
                or not SHA256_PATTERN.fullmatch(str(public_identity.get("sha256", "")))
                or public_identity.get("residual_absolute_path_count") != 0
                or not isinstance(tensor, dict)
                or tensor.get("algorithm")
                != "recursive_path_dtype_shape_raw_bytes_sha256_v1"
                or isinstance(tensor.get("tensor_count"), bool)
                or not isinstance(tensor.get("tensor_count"), int)
                or tensor["tensor_count"] <= 0
                or not SHA256_PATTERN.fullmatch(str(tensor.get("sha256", "")))
            ):
                raise CoreAuditError(
                    f"Pinned public identity is malformed: {experiment_id}"
                )
    if spec["status"] == "public_identities_pinned":
        expected_public_total = sum(
            int(row["public_identity"]["bytes"]) for row in spec_records
        )
        if spec.get("public_total_bytes") != expected_public_total:
            raise CoreAuditError("Pinned public byte total is inconsistent")


def _resolve_protected_file(project_root: Path, relative: str) -> Path:
    relative = _safe_relative_path(relative, "historical checkpoint path")
    candidate = project_root.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise CoreAuditError(f"Historical checkpoint must not be a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise CoreAuditError(f"Invalid historical checkpoint path: {relative}") from error
    if not resolved.is_file():
        raise CoreAuditError(f"Historical checkpoint is not a regular file: {relative}")
    return resolved


def _source_identity(path: Path, relative: str) -> dict:
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _metadata_relative_path(path: Path, source_project: Path, fallback: str) -> str:
    """Name trusted metadata without exposing either checkout's absolute path."""

    for root in (PROJECT_ROOT.resolve(), source_project):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return fallback


def _verify_historical_identity(path: Path, record: Mapping[str, Any]) -> dict:
    relative = str(record["historical_project_relative_path"])
    actual = _source_identity(path, relative)
    expected = {
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
    }
    mismatches = {
        key: {"actual": actual[key], "index": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise CoreAuditError(
            f"Historical checkpoint identity mismatch for {record['experiment_id']}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise CoreAuditError("PyTorch is required for the checkpoint audit") from error
    return torch


def _serialization_runtime(torch: Any) -> dict:
    """Return only the two versions that can affect ``torch.save`` bytes."""

    values = {
        "python": platform.python_version(),
        "torch": str(getattr(torch, "__version__", "")),
    }
    safe_version = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}$")
    invalid = {
        key: value for key, value in values.items() if not safe_version.fullmatch(value)
    }
    if invalid:
        raise CoreAuditError("Serialization runtime has an invalid version string")
    return values


def _restricted_load(source: Any, torch: Any) -> Any:
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CoreAuditError(
            f"Restricted checkpoint deserialization failed: {type(error).__name__}: {error}"
        ) from error


def _require_tensor_state(value: Any, label: str, torch: Any) -> dict:
    if not isinstance(value, dict) or not value:
        raise CoreAuditError(f"{label} must be a non-empty tensor dictionary")
    invalid = [
        key
        for key, tensor in value.items()
        if not isinstance(key, str) or not torch.is_tensor(tensor)
    ]
    if invalid:
        raise CoreAuditError(f"{label} contains non-tensor entries")
    return value


def _state_summary(state: Mapping[str, Any]) -> dict:
    keys = sorted(state)
    return {
        "tensor_count": len(keys),
        "tensor_elements": sum(int(tensor.numel()) for tensor in state.values()),
        "tensor_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in state.values()
        ),
        "lora_A_tensor_count": sum(key.endswith(".lora_A") for key in keys),
        "lora_B_tensor_count": sum(key.endswith(".lora_B") for key in keys),
        "task_head_tensor_count": sum(
            "score_head" in key or "class_embed" in key for key in keys
        ),
    }


def _validate_factor_state(
    state: Mapping[str, Any], *, allow_a: bool, allow_b: bool, require_head: bool,
    label: str,
) -> dict:
    summary = _state_summary(state)
    if allow_a != (summary["lora_A_tensor_count"] > 0):
        raise CoreAuditError(f"{label} has an unexpected LoRA-A inventory")
    if allow_b != (summary["lora_B_tensor_count"] > 0):
        raise CoreAuditError(f"{label} has an unexpected LoRA-B inventory")
    if require_head and summary["task_head_tensor_count"] <= 0:
        raise CoreAuditError(f"{label} has no task-head tensor")
    return summary


def _expected_compatibility(seed: int, method: str) -> dict:
    uses_lora = method != "full_ft"
    return {
        "fl_method": method,
        "model_name": "rtdetr-l",
        "num_classes": 4,
        "num_clients": 3,
        "fl_rounds": 20,
        "local_epochs": 5,
        "seed": seed,
        "partition_seed": seed,
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "lora_rank": 8 if uses_lora else None,
        "lora_alpha": 16.0 if uses_lora else None,
        "lora_dropout": 0.0 if uses_lora else None,
        "apply_lora_backbone": True if uses_lora else None,
        "apply_lora_decoder": True if uses_lora else None,
        "backbone_min_channels": 64 if uses_lora else None,
    }


def validate_checkpoint_contract(
    payload: Any, record: Mapping[str, Any], torch: Any
) -> dict:
    """Validate the release-relevant schema and method-specific state layout."""

    if not isinstance(payload, dict):
        raise CoreAuditError("Checkpoint payload is not a dictionary")
    method = str(record["method"])
    seed = int(record["training_seed"])
    contract = METHOD_CONTRACTS[method]
    checks = {
        "schema_version": (payload.get("schema_version"), 5),
        "checkpoint_kind": (payload.get("checkpoint_kind"), "federated_personalized"),
        "resume_capability": (
            payload.get("resume_capability"),
            "evaluation_only_no_optimizer_scheduler_or_rng_state",
        ),
        "selection": (payload.get("selection"), "best_macro_client_local_val_AP"),
        "round": (payload.get("round"), int(record["selected_at"])),
        "best_round": (payload.get("best_round"), int(record["selected_at"])),
        "training_rounds_executed": (payload.get("training_rounds_executed"), 20),
        "fl_method": (payload.get("fl_method"), method),
        "num_clients": (payload.get("num_clients"), 3),
        "class_names": (payload.get("class_names"), CLASS_NAMES),
        "split_manifest_sha256": (
            payload.get("split_manifest_sha256"), record["split_manifest_sha256"]
        ),
        "nonfloating_state_policy": (
            payload.get("nonfloating_state_policy"), "retain_previous_server_value"
        ),
        "federated_payload_policy": (
            payload.get("federated_payload_policy"), contract["payload_policy"]
        ),
        "shared_lora_factor_role": (
            payload.get("shared_lora_factor_role"), contract["shared_role"]
        ),
        "local_lora_factor_role": (
            payload.get("local_lora_factor_role"), contract["local_role"]
        ),
    }
    mismatches = {
        key: {"checkpoint": left, "required": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    if mismatches:
        raise CoreAuditError(
            f"Checkpoint contract mismatch for {record['experiment_id']}: "
            + json.dumps(mismatches, sort_keys=True)
        )

    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, dict):
        raise CoreAuditError("Checkpoint has no compatibility manifest")
    expected_compatibility = _expected_compatibility(seed, method)
    compatibility_mismatches = {
        key: {"checkpoint": compatibility.get(key), "required": value}
        for key, value in expected_compatibility.items()
        if compatibility.get(key) != value
    }
    if compatibility_mismatches:
        raise CoreAuditError(
            f"Compatibility mismatch for {record['experiment_id']}: "
            + json.dumps(compatibility_mismatches, sort_keys=True)
        )

    shared = _require_tensor_state(payload.get("shared_state"), "shared_state", torch)
    local_states = payload.get("local_personalized_states")
    if method == "full_ft":
        shared_summary = _state_summary(shared)
        if local_states is not None:
            raise CoreAuditError("Full FT checkpoint unexpectedly has local states")
        local_summaries: list[dict] = []
    elif method == "lora":
        shared_summary = _validate_factor_state(
            shared, allow_a=True, allow_b=True, require_head=True,
            label="FedLoRA-AB shared_state",
        )
        if local_states is not None:
            raise CoreAuditError("FedLoRA-AB checkpoint unexpectedly has local states")
        local_summaries = []
    else:
        shared_a = method == "fedsa_lora"
        shared_summary = _validate_factor_state(
            shared,
            allow_a=shared_a,
            allow_b=not shared_a,
            require_head=True,
            label=f"{method} shared_state",
        )
        if not isinstance(local_states, list) or len(local_states) != 3:
            raise CoreAuditError(
                f"{method} checkpoint must contain three personalized states"
            )
        local_summaries = []
        for client_id, state in enumerate(local_states):
            checked = _require_tensor_state(
                state, f"local_personalized_states[{client_id}]", torch
            )
            local_summaries.append(
                _validate_factor_state(
                    checked,
                    allow_a=not shared_a,
                    allow_b=shared_a,
                    require_head=False,
                    label=f"{method} client {client_id} local state",
                )
            )

    return {
        "schema_version": 5,
        "checkpoint_kind": "federated_personalized",
        "selection": "best_macro_client_local_val_AP",
        "selected_round": int(record["selected_at"]),
        "training_rounds_executed": 20,
        "federated_payload_policy": contract["payload_policy"],
        "shared_lora_factor_role": contract["shared_role"],
        "local_lora_factor_role": contract["local_role"],
        "shared_state": shared_summary,
        "local_personalized_states": local_summaries,
    }


def _discover_public_identity(payload: dict, torch: Any) -> dict:
    historical_paths = [
        value
        for _, value in release_builder._walk_strings(payload)
        if release_builder._path_string(value)
    ]
    public_payload, replacements = release_builder.sanitize_checkpoint_payload(payload)
    historical_fingerprint, historical_count = release_builder._tensor_fingerprint(
        payload, torch
    )
    public_fingerprint, public_count = release_builder._tensor_fingerprint(
        public_payload, torch
    )
    if (public_fingerprint, public_count) != (
        historical_fingerprint,
        historical_count,
    ):
        raise CoreAuditError("Tensor fingerprint changed during path sanitization")

    buffer = io.BytesIO()
    torch.save(public_payload, buffer)
    raw = buffer.getvalue()
    reloaded = _restricted_load(io.BytesIO(raw), torch)
    release_builder._assert_payload_equivalent(payload, reloaded, torch)
    reloaded_fingerprint, reloaded_count = release_builder._tensor_fingerprint(
        reloaded, torch
    )
    if (reloaded_fingerprint, reloaded_count) != (
        historical_fingerprint,
        historical_count,
    ):
        raise CoreAuditError("Serialized public candidate changed tensor state")
    if any(
        release_builder._path_string(value)
        for _, value in release_builder._walk_strings(reloaded)
    ):
        raise CoreAuditError("Serialized public candidate contains an absolute path")
    release_builder._assert_no_private_checkpoint_markers(raw, historical_paths)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_fingerprint": {
            "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
            "tensor_count": historical_count,
            "sha256": historical_fingerprint,
        },
        "path_replacements": replacements,
        "residual_absolute_path_count": 0,
    }


def _assert_existing_representative_identity(
    record: Mapping[str, Any], public_identity: Mapping[str, Any]
) -> None:
    """Require the already-published seed-42 FedLoRA-A identity to reproduce."""

    if record["experiment_id"] != release_builder.evaluation.TARGET_EXPERIMENT_ID:
        return
    fingerprint = public_identity.get("tensor_fingerprint")
    if not isinstance(fingerprint, dict):
        raise CoreAuditError("Representative public candidate has no fingerprint")
    checks = {
        "bytes": (
            public_identity.get("bytes"),
            release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_BYTES,
        ),
        "sha256": (
            public_identity.get("sha256"),
            release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_SHA256,
        ),
        "tensor_fingerprint.tensor_count": (
            fingerprint.get("tensor_count"),
            release_builder.evaluation.FROZEN_PUBLIC_TENSOR_COUNT,
        ),
        "tensor_fingerprint.sha256": (
            fingerprint.get("sha256"),
            release_builder.evaluation.FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256,
        ),
    }
    mismatches = {
        key: {"discovered": left, "published": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    if mismatches:
        raise CoreAuditError(
            "Seed-42 FedLoRA-A candidate does not reproduce the published "
            "representative checkpoint identity: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _walk_report_strings(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterable[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield path + (f"<dict-key:{key}>",), key
            yield from _walk_report_strings(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_report_strings(item, path + (index,))
    elif isinstance(value, str):
        yield path, value


def _assert_path_free_report(report: Mapping[str, Any]) -> None:
    reviewed_json_pointers = {
        "/architecture/model_weight_path",
        "/experiment/model_weights",
    }
    leaked = []
    for path, value in _walk_report_strings(report):
        if (
            path
            and path[-1] == "json_pointer"
            and value in reviewed_json_pointers
        ):
            continue
        if release_builder._path_string(value):
            leaked.append({
                "report_pointer": "/" + "/".join(map(str, path)),
                "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            })
    if leaked:
        raise CoreAuditError(
            "Audit report contains an absolute local filesystem path: "
            + json.dumps(leaked, sort_keys=True)
        )


def _write_exclusive(path: Path, raw: bytes) -> None:
    if not path.parent.is_dir():
        raise CoreAuditError(f"Output parent does not exist: {path.parent}")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise CoreAuditError(f"Cannot create audit report: {error}") from error


def run_audit(
    project_dir: Path, index_path: Path, spec_path: Path, output: Path
) -> dict:
    project_root = project_dir.expanduser().resolve(strict=True)
    index_input = index_path.expanduser()
    if index_input.is_symlink():
        raise CoreAuditError("Checkpoint index must not be a symlink")
    try:
        index_resolved = index_input.resolve(strict=True)
    except OSError as error:
        raise CoreAuditError(f"Missing checkpoint index: {index_path}") from error
    if not index_resolved.is_file():
        raise CoreAuditError("Checkpoint index is not a regular file")

    spec_input = spec_path.expanduser()
    if spec_input.is_symlink():
        raise CoreAuditError("Paper-core release spec must not be a symlink")
    try:
        spec_resolved = spec_input.resolve(strict=True)
    except OSError as error:
        raise CoreAuditError(f"Missing paper-core release spec: {spec_path}") from error
    if not spec_resolved.is_file():
        raise CoreAuditError("Paper-core release spec is not a regular file")

    try:
        records = load_index(index_resolved)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise CoreAuditError(f"Invalid checkpoint index: {error}") from error
    selected = select_core_records(records)
    index_relative = _metadata_relative_path(
        index_resolved, project_root, "checkpoint_index.json"
    )
    index_before = _source_identity(index_resolved, index_relative)
    spec_relative = _metadata_relative_path(
        spec_resolved, project_root, "paper_core_checkpoint_release_spec.json"
    )
    spec_before = _source_identity(spec_resolved, spec_relative)
    spec = _load_release_spec(spec_resolved)
    _validate_spec_against_index(spec, selected, index_before)

    sources: list[tuple[Path, dict, dict]] = []
    for record in selected:
        path = _resolve_protected_file(
            project_root, str(record["historical_project_relative_path"])
        )
        sources.append((path, record, _verify_historical_identity(path, record)))

    torch = _load_torch()
    runtime = _serialization_runtime(torch)
    if (
        spec["status"] == "public_identities_pinned"
        and spec.get("serialization_runtime") != runtime
    ):
        raise CoreAuditError(
            "Pinned public identities require the audited serialization runtime"
        )
    spec_by_experiment = {
        row["experiment_id"]: row for row in spec["records"]
    }
    audited_records = []
    for path, record, before in sources:
        payload = _restricted_load(str(path), torch)
        contract = validate_checkpoint_contract(payload, record, torch)
        public_identity = _discover_public_identity(payload, torch)
        _assert_existing_representative_identity(record, public_identity)
        if spec["status"] == "public_identities_pinned":
            expected_public = spec_by_experiment[record["experiment_id"]][
                "public_identity"
            ]
            if public_identity != expected_public:
                raise CoreAuditError(
                    "Discovered public identity differs from the pinned spec for "
                    f"{record['experiment_id']}"
                )
        audited_records.append({
            "experiment_id": record["experiment_id"],
            "method": record["method"],
            "training_seed": record["training_seed"],
            "partition_seed": record["partition_seed"],
            "partition": record["partition"],
            "rank": record["rank"],
            "historical_checkpoint": {
                **before,
                "release_asset_name": record["release_asset_name"],
            },
            "checkpoint_contract": contract,
            "public_checkpoint_candidate": {
                "file_name": record["release_asset_name"],
                "release_asset_name": record["release_asset_name"],
                **public_identity,
            },
        })
        del payload
        gc.collect()

    protected_records = []
    for path, record, before in sources:
        after = _source_identity(path, before["relative_path"])
        identical = before == after
        protected_records.append({
            "relative_path": before["relative_path"],
            "bytes": before["bytes"],
            "sha256": before["sha256"],
            "before_after_identical": identical,
        })
        if not identical:
            raise CoreAuditError(
                f"Protected checkpoint changed during audit: {record['experiment_id']}"
            )
    index_after = _source_identity(index_resolved, index_relative)
    if index_before != index_after:
        raise CoreAuditError("Checkpoint index changed during audit")
    protected_records.append({
        "relative_path": index_before["relative_path"],
        "bytes": index_before["bytes"],
        "sha256": index_before["sha256"],
        "before_after_identical": True,
    })
    spec_after = _source_identity(spec_resolved, spec_relative)
    if spec_before != spec_after:
        raise CoreAuditError("Paper-core release spec changed during audit")
    protected_records.append({
        "relative_path": spec_before["relative_path"],
        "bytes": spec_before["bytes"],
        "sha256": spec_before["sha256"],
        "before_after_identical": True,
    })

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "policy": AUDIT_POLICY,
        "scope": {
            "training_and_partition_seeds": list(SEEDS),
            "methods": list(METHOD_ORDER),
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "lora_rank": 8,
            "lora_targets": "backbone_plus_decoder",
            "record_count": 12,
        },
        "checkpoint_index": {
            "relative_path": index_before["relative_path"],
            "bytes": index_before["bytes"],
            "sha256": index_before["sha256"],
        },
        "release_spec": {
            "release_id": spec["release_id"],
            "relative_path": spec_before["relative_path"],
            "bytes": spec_before["bytes"],
            "sha256": spec_before["sha256"],
        },
        "serialization_runtime": runtime,
        "records": audited_records,
        "protected_source_gate": {
            "status": "pass",
            "protected_file_count": len(protected_records),
            "all_before_after_sha256_identical": True,
            "files": protected_records,
        },
        "totals": {
            "historical_checkpoint_bytes": sum(
                row["historical_checkpoint"]["bytes"] for row in audited_records
            ),
            "public_candidate_bytes": sum(
                row["public_checkpoint_candidate"]["bytes"]
                for row in audited_records
            ),
        },
    }
    _assert_path_free_report(report)
    _write_exclusive(output.expanduser().resolve(), _canonical_json_bytes(report))
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint_index.json",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.project_dir, args.index, args.spec, args.output)
    except (CoreAuditError, OSError, ValueError, TypeError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print(
        "[PASS] paper-core checkpoint audit: "
        f"{len(report['records'])}/12 inputs; "
        f"protected_files={report['protected_source_gate']['protected_file_count']}"
    )
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
