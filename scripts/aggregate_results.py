#!/usr/bin/env python3
"""Recursively aggregate paired-seed experiment results into publication tables."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

METRICS = (
    "macro_AP", "macro_AP50", "macro_AP75",
    "client_sd_AP", "client_sd_AP50", "client_sd_AP75",
    "worst_AP", "worst_AP50", "worst_AP75",
    "common_AP", "common_AP50", "common_AP75",
    "mia_auc", "mia_tpr_at_1fpr", "mia_asr",
)

PROTOCOL_DROP_KEYS = {
    "seed", "partition_seed", "cross_client_eval", "model_weights",
}

PAIRED_METRICS = (
    "macro_AP", "macro_AP50", "macro_AP75",
    "client_sd_AP", "client_sd_AP50", "client_sd_AP75",
    "worst_AP", "worst_AP50", "worst_AP75", "common_AP",
)
LOWER_IS_BETTER = {"client_sd_AP", "client_sd_AP50", "client_sd_AP75"}
FACTOR_SHARING_PAIRED_METRICS = PAIRED_METRICS + (
    "communication_params", "one_client_one_way_mb", "cumulative_total_mb",
)
FACTOR_SHARING_LOWER_IS_BETTER = LOWER_IS_BETTER | {
    "communication_params", "one_client_one_way_mb", "cumulative_total_mb",
}
METHOD_SPECIFIC_PROTOCOL_KEYS = {
    "fl_method", "lora_rank", "lora_alpha", "lora_dropout",
    "apply_lora_backbone", "apply_lora_decoder", "backbone_min_channels",
    "lr", "head_lr", "backbone_lr_ratio",
}
SPLIT_GENERATION_PROTOCOL_KEYS = (
    "schema_version",
    "partition",
    "dirichlet_alpha",
    "partition_algorithm",
    "num_clients",
    "min_bbox_area",
    "min_bbox_side",
    "drop_empty_images",
    "crowd_policy",
    "category_policy",
    "source_category_audit",
    "source_split_policy",
    "official_split_preserved",
    "source_identity_policy",
    "source_split_priority",
    "client_partition_unit",
    "raw_source_counts",
    "source_split_counts",
    "official_count_gate",
    "class_names",
    "cat_id_to_label",
    "split_counts",
    "annotation_sha256",
    "cross_split_source_audit",
    "post_policy_cross_split_source_check",
    "image_hash_check_enabled",
    "image_decode_check",
    "cross_split_client_source_group_check",
)

EXPECTED_SPLIT_SCHEMA_VERSION = 7
OFFICIAL_SOURCE_SPLIT_POLICY = "official_aod4_v6"
EXCLUSIVE_SOURCE_SPLIT_POLICY = "exclusive_highest_evaluation_priority"
SUPPORTED_SOURCE_SPLIT_POLICIES = {
    OFFICIAL_SOURCE_SPLIT_POLICY,
    EXCLUSIVE_SOURCE_SPLIT_POLICY,
}
EXPECTED_SOURCE_IDENTITY_POLICY = (
    "roboflow_source_key_or_exact_sha256_connected_components"
)
EXPECTED_CLIENT_PARTITION_UNIT = "source_group"
EXPECTED_SOURCE_SPLIT_PRIORITY = ["test", "val", "train"]
EXPECTED_CROWD_POLICY = (
    "require_zero_crowd_annotations_for_YOLO_metric_equivalence"
)
EXPECTED_CATEGORY_POLICY = (
    "exact_aod4_targets_ignore_only_unreferenced_declared_categories"
)
EXPECTED_IID_PARTITION_ALGORITHM = (
    "random_source_group_lpt_balance_cross_split_owner_v2"
)
EXPECTED_DIRICHLET_PARTITION_ALGORITHM = (
    "source_group_target_deficit_balance_cross_split_owner_v2"
)


def _number(value, default=float("nan")):
    """Return a finite float, including when a legacy fallback is ``None``."""
    for candidate in (value, default):
        try:
            result = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return float("nan")


def _integer(value, default: int) -> int:
    number = _number(value, default)
    return int(number) if math.isfinite(number) else int(default)


def _nested(payload, *keys, default=None):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _mia_value(result: dict, key: str):
    value = _nested(result, "mia", "macro", key, "macro_mean")
    if value is None:
        value = _nested(result, "mia", key)
    return _number(value)


def _communication_value(result: dict, *candidate_paths):
    for path in candidate_paths:
        value = _nested(result, *path)
        if value is not None:
            return _number(value)
    return float("nan")


def _normalized_split_digest(result: dict):
    """Normalize standalone and federated names for the same manifest digest."""
    digest = result.get("split_manifest_sha256")
    if digest is None:
        digest = result.get("split_file_sha256")
    if digest is None:
        return None
    return str(digest).strip().lower()


def _load_recorded_split_metadata(result: dict) -> dict:
    """Read immutable split-generation metadata without retaining its realization."""
    metadata = result.get("split_metadata")
    if isinstance(metadata, dict) and metadata:
        return metadata
    split_file = result.get("split_file")
    if not split_file or not os.path.isfile(str(split_file)):
        return {}
    try:
        with open(str(split_file), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _split_generation_protocol(result: dict) -> dict:
    """Return seed-invariant data-generation and preprocessing policy.

    The immutable manifest digest, partition seed, sampled Dirichlet proportions,
    client assignments/statistics and generated YOLO-tree digest are deliberately
    absent. Dataset identity, algorithm, filtering, class mapping and verification
    policy remain, so only genuine replicate realizations share a family.
    """
    metadata = _load_recorded_split_metadata(result)
    protocol = {key: metadata.get(key) for key in SPLIT_GENERATION_PROTOCOL_KEYS}
    protocol["partition"] = metadata.get("partition", result.get("partition"))
    protocol["dirichlet_alpha"] = metadata.get(
        "dirichlet_alpha", result.get("dirichlet_alpha")
    )
    protocol["num_clients"] = metadata.get("num_clients", result.get("num_clients"))
    # The cross-split ownership report also stores seed-dependent move lists and
    # realized client counts.  Keep only its invariant policy/audit fields in
    # the cross-seed protocol fingerprint; the immutable split digest validates
    # each realization inside a paired seed.
    cross_client = metadata.get("cross_split_client_source_group_check")
    if isinstance(cross_client, dict):
        protocol["cross_split_client_source_group_check"] = {
            "schema_version": cross_client.get("schema_version"),
            "policy": cross_client.get("policy"),
            "shared_source_groups": cross_client.get("shared_source_groups"),
            "cross_split_client_owner_conflicts": cross_client.get(
                "cross_split_client_owner_conflicts"
            ),
        }
    # The schema-v7 inventory contains one SHA-256 record per raw image and is
    # intentionally not copied into the protocol fingerprint.  Its compact
    # commitments are sufficient to distinguish source datasets without
    # inflating every aggregate group key by tens of thousands of records.
    source_inventory = metadata.get("source_hash_inventory")
    if isinstance(source_inventory, dict):
        protocol["source_hash_inventory_summary"] = {
            "schema_version": source_inventory.get("schema_version"),
            "identity_policy": source_inventory.get("identity_policy"),
            "inventory_sha256": source_inventory.get("inventory_sha256"),
            "per_split_image_tree_sha256": source_inventory.get(
                "per_split_image_tree_sha256"
            ),
        }
    else:
        protocol["source_hash_inventory_summary"] = None
    protocol["metadata_recorded"] = bool(metadata)
    return protocol


def _sha256_of_json(payload) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mia_protocol_manifest(result: dict, experiment_manifest: dict) -> dict:
    """Capture common attack configuration without client-specific realizations.

    Older Solo/Centralized results did not copy the configured MIA arguments into
    ``training_experiment``. For those files, common configuration values can be
    recovered from the attack record. Client IDs and realized sample counts must
    not enter this signature: Solo emits one file per client, so those values are
    expected to differ before the client files are collapsed into one run.
    """
    protocol: dict = {
        "configured_max_samples": experiment_manifest.get("mia_max_samples"),
        "configured_calibration_fraction": experiment_manifest.get(
            "mia_calibration_fraction"
        ),
        "configured_attack": experiment_manifest.get("mia_attack"),
        "configured_nonmember_source_policy": experiment_manifest.get(
            "mia_nonmember_source_policy"
        ),
        "configured_bootstrap": experiment_manifest.get("mia_bootstrap"),
    }
    mia = result.get("mia")
    if not isinstance(mia, dict):
        return protocol

    configuration = mia.get("configuration")
    if (
        protocol["configured_nonmember_source_policy"] is None
        and isinstance(configuration, dict)
    ):
        protocol["configured_nonmember_source_policy"] = configuration.get(
            "nonmember_source_policy"
        )

    records = mia.get("per_client", [])
    if not isinstance(records, list):
        records = []
    attacks = set()
    calibration_fractions = set()
    bootstrap_specs = set()
    for entry in records:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", entry)
        if not isinstance(metrics, dict):
            continue
        bootstrap = metrics.get("bootstrap_95_ci", metrics.get("bootstrap", {}))
        if not isinstance(bootstrap, dict):
            bootstrap = {}
        if metrics.get("attack") is not None:
            attacks.add(str(metrics["attack"]))
        calibration = _number(metrics.get("requested_calibration_fraction"))
        if math.isfinite(calibration):
            calibration_fractions.add(calibration)
        bootstrap_spec = (
            bootstrap.get("method"),
            bootstrap.get("resamples"),
            bootstrap.get("confidence_level"),
        )
        if any(value is not None for value in bootstrap_spec):
            bootstrap_specs.add(bootstrap_spec)

    # Only fill fields absent from a modern experiment manifest. These are
    # attack configuration values shared across clients, never measured counts.
    if protocol["configured_calibration_fraction"] is None and calibration_fractions:
        protocol["configured_calibration_fraction"] = sorted(calibration_fractions)
    if protocol["configured_bootstrap"] is None and bootstrap_specs:
        protocol["configured_bootstrap"] = [
            {"method": method, "resamples": resamples, "confidence_level": confidence}
            for method, resamples, confidence in sorted(
                bootstrap_specs, key=lambda item: tuple(str(value) for value in item)
            )
        ]
    if protocol["configured_attack"] is None and attacks:
        protocol["configured_attack"] = sorted(attacks)
    return protocol


def _protocol_signature(
    result: dict, split_generation_protocol: Optional[dict] = None
) -> tuple[str, dict]:
    manifest = result.get("training_experiment")
    if not isinstance(manifest, dict) or not manifest:
        manifest = result.get("experiment")
    if not isinstance(manifest, dict) or not manifest:
        manifest = {
            "legacy_result": True,
            "mode": result.get("mode"),
            "fl_method": result.get("fl_method"),
            "partition": result.get("partition"),
            "dirichlet_alpha": result.get("dirichlet_alpha"),
            "lora_rank": result.get("lora_rank"),
            "lora_alpha": result.get("lora_alpha"),
            "num_clients": result.get("num_clients"),
        }
    canonical = {
        str(key): value for key, value in manifest.items()
        if key not in PROTOCOL_DROP_KEYS
    }
    # The actual split digest is validated within each seed pair, but excluded
    # here so independent partition-seed realizations form a cross-seed family.
    canonical["split_generation_protocol"] = (
        split_generation_protocol
        if split_generation_protocol is not None
        else _split_generation_protocol(result)
    )
    canonical["mia_protocol"] = _mia_protocol_manifest(result, manifest)
    architecture = result.get("architecture", {})
    if isinstance(architecture, dict):
        canonical["ultralytics_version"] = architecture.get("ultralytics_version")
        canonical["pretrained_tensor_state_sha256"] = architecture.get(
            "pretrained_tensor_state_sha256"
        )
    return _sha256_of_json(canonical), canonical


def normalize_result(path: str, result: dict) -> dict:
    """Normalize schema-v2 and older JSONs without silently inventing values."""
    method = str(result.get("method", f"{result.get('mode', 'unknown')}-{result.get('fl_method', '')}"))
    mode = str(result.get("mode", "fl" if method.startswith("FL") else "unknown"))
    split_generation_protocol = _split_generation_protocol(result)
    split_generation_signature = _sha256_of_json(split_generation_protocol)
    protocol_signature, protocol_manifest = _protocol_signature(
        result, split_generation_protocol
    )
    row = {
        "path": os.path.abspath(path),
        "method": method,
        "mode": mode,
        "fl_method": result.get("fl_method"),
        "seed": _integer(result.get("seed"), -1),
        "partition_seed": _integer(
            result.get("partition_seed"), _integer(result.get("seed"), -1)
        ),
        "partition": result.get("partition", "dirichlet"),
        "dirichlet_alpha": result.get("dirichlet_alpha"),
        "lora_rank": result.get("lora_rank"),
        "lora_alpha": result.get("lora_alpha"),
        "apply_lora_backbone": result.get("apply_lora_backbone"),
        "apply_lora_decoder": result.get("apply_lora_decoder"),
        "num_clients": _integer(result.get("num_clients"), 1),
        "client_id": result.get("client_id"),
        "split_manifest_sha256": _normalized_split_digest(result),
        "split_generation_signature": split_generation_signature,
        "split_generation_protocol": split_generation_protocol,
        "protocol_signature": protocol_signature,
        "protocol_manifest": protocol_manifest,
    }

    client_summary = result.get("client_summary", {})
    if client_summary:
        for metric in ("AP", "AP50", "AP75"):
            row[f"macro_{metric}"] = _number(_nested(client_summary, metric, "macro_mean"))
            sd_value = _nested(client_summary, metric, "client_sample_sd")
            if sd_value is None:
                sd_value = _nested(client_summary, metric, "sample_std")
            worst_value = _nested(client_summary, metric, "worst_client")
            if worst_value is None:
                worst_value = _nested(client_summary, metric, "worst")
            row[f"client_sd_{metric}"] = _number(sd_value)
            row[f"worst_{metric}"] = _number(worst_value)
    elif "own_client_test" in result:
        own = result["own_client_test"]
        for metric in ("AP", "AP50", "AP75"):
            row[f"macro_{metric}"] = _number(own.get(metric))
            row[f"client_sd_{metric}"] = float("nan")
            row[f"worst_{metric}"] = _number(own.get(metric))
    else:
        # Compatibility with the original flat result schema.
        for metric in ("AP", "AP50", "AP75"):
            row[f"macro_{metric}"] = _number(result.get(f"avg_test_{metric}"))
            row[f"client_sd_{metric}"] = _number(result.get(f"std_test_{metric}"))
            row[f"worst_{metric}"] = float("nan")

    common = result.get("common_test", result.get("test_metrics", {}))
    for metric in ("AP", "AP50", "AP75"):
        common_value = common.get(metric) if isinstance(common, dict) else None
        if common_value is None:
            common_value = _nested(
                common, "summary_across_personalized_models", metric, "macro_mean"
            )
        row[f"common_{metric}"] = _number(common_value)

    row.update({
        "mia_auc": _mia_value(result, "auc_roc"),
        "mia_tpr_at_1fpr": _mia_value(result, "tpr_at_1fpr"),
        "mia_asr": _mia_value(result, "asr"),
        "total_params": _number(
            _nested(result, "parameter_counts", "total_params"), result.get("total_params")
        ),
        "trainable_params": _number(
            _nested(result, "parameter_counts", "trainable_params"),
            result.get("trainable_params"),
        ),
        "communication_params": _number(
            _nested(result, "parameter_counts", "communication_params"),
            result.get("comm_params"),
        ),
        "trainable_ratio_pct": _number(
            _nested(result, "parameter_counts", "trainable_ratio_pct")
        ),
        "parameter_saving_pct": _number(
            _nested(result, "parameter_counts", "parameter_saving_pct"),
            result.get("param_efficiency"),
        ),
        "communication_saving_pct": _communication_value(
            result,
            ("communication", "saving_vs_full_ft_pct"),
            ("parameter_counts", "communication_saving_pct"),
        ),
        "communication_byte_saving_pct": _communication_value(
            result,
            ("communication", "byte_saving_vs_full_ft_pct"),
            ("communication", "saving_vs_full_ft_pct"),
        ),
        "communication_element_saving_pct": _communication_value(
            result,
            ("communication", "element_saving_vs_full_ft_pct"),
            ("parameter_counts", "communication_saving_pct"),
        ),
        "one_client_one_way_mb": _communication_value(
            result,
            ("communication", "one_client_one_way", "mb"),
            ("communication", "one_client_one_way_mb"),
            ("communication", "one_way_client_mb"),
        ),
        "system_round_total_mb": _communication_value(
            result,
            ("communication", "round_total", "mb"),
            ("communication", "system_round_total_mb"),
            ("communication", "round_total_mb"),
            ("per_round_comm_mb",),
        ),
        "cumulative_total_mb": _communication_value(
            result,
            ("communication", "cumulative_total", "mb"),
            ("communication", "cumulative_total_mb"),
            ("communication", "total_mb"),
            ("total_comm_mb",),
        ),
    })
    communication_block = result.get("communication")
    if isinstance(communication_block, dict) and communication_block.get("applicable") is False:
        for key in (
            "communication_params", "communication_saving_pct",
            "communication_byte_saving_pct", "communication_element_saving_pct",
            "one_client_one_way_mb",
            "system_round_total_mb", "cumulative_total_mb",
        ):
            row[key] = float("nan")
    row["total_params_m"] = row["total_params"] / 1_000_000.0
    row["trainable_params_m"] = row["trainable_params"] / 1_000_000.0
    row["communication_params_m"] = row["communication_params"] / 1_000_000.0
    return row


def _group_key(row: dict):
    return (
        row["method"], row["fl_method"], row["partition"], row["dirichlet_alpha"],
        row["lora_rank"], row["lora_alpha"], row["apply_lora_backbone"],
        row["apply_lora_decoder"], row["num_clients"],
        row["protocol_signature"],
    )


def _validate_split_digest_consistency(rows: list[dict]) -> None:
    """Require one immutable split per data protocol and paired replicate.

    Different partition seeds are valid cross-seed replicates and therefore may
    have different digests. Within one ``(seed, partition_seed)`` replicate,
    however, Local/Centralized/FL methods using the same generation protocol must
    all reference exactly the same 64-character SHA-256 digest.
    """
    required_generation_fields = set(SPLIT_GENERATION_PROTOCOL_KEYS) - {
        "dirichlet_alpha"
    }
    missing_generation_protocol = []
    for row in rows:
        protocol = row.get("split_generation_protocol", {})
        missing = sorted(
            key for key in required_generation_fields if protocol.get(key) is None
        )
        if not protocol.get("metadata_recorded") or missing:
            missing_generation_protocol.append({
                "path": row["path"],
                "metadata_recorded": protocol.get("metadata_recorded"),
                "missing_fields": missing,
            })
    if missing_generation_protocol:
        raise ValueError(
            "Publication aggregation requires recorded split-generation metadata "
            "(keep each result's split_file accessible or embed split_metadata):\n"
            + json.dumps(missing_generation_protocol[:10], indent=2, ensure_ascii=False)
        )

    invalid_schema7_protocol = []
    for row in rows:
        protocol = row["split_generation_protocol"]
        summary = protocol.get("source_hash_inventory_summary")
        tree_hashes = (
            summary.get("per_split_image_tree_sha256")
            if isinstance(summary, dict) else None
        )
        post_policy = protocol.get("post_policy_cross_split_source_check")
        source_audit = protocol.get("cross_split_source_audit")
        reasons = []
        if protocol.get("schema_version") != EXPECTED_SPLIT_SCHEMA_VERSION:
            reasons.append("schema_version!=7")
        source_split_policy = protocol.get("source_split_policy")
        if source_split_policy not in SUPPORTED_SOURCE_SPLIT_POLICIES:
            reasons.append("unexpected source_split_policy")
        if protocol.get("official_split_preserved") != (
            source_split_policy == OFFICIAL_SOURCE_SPLIT_POLICY
        ):
            reasons.append("official_split_preserved marker mismatch")
        if protocol.get("source_identity_policy") != EXPECTED_SOURCE_IDENTITY_POLICY:
            reasons.append("unexpected source_identity_policy")
        if protocol.get("client_partition_unit") != EXPECTED_CLIENT_PARTITION_UNIT:
            reasons.append("client_partition_unit!=source_group")
        cross_client_check = protocol.get("cross_split_client_source_group_check")
        if not (
            isinstance(cross_client_check, dict)
            and cross_client_check.get("policy")
            == "train_then_val_then_test_global_source_group_client_owner_v1"
            and cross_client_check.get("cross_split_client_owner_conflicts") == 0
        ):
            reasons.append("invalid cross-split source-group client ownership")
        expected_priority = (
            [] if source_split_policy == OFFICIAL_SOURCE_SPLIT_POLICY
            else EXPECTED_SOURCE_SPLIT_PRIORITY
        )
        if protocol.get("source_split_priority") != expected_priority:
            reasons.append("source_split_priority inconsistent with policy")
        if protocol.get("image_hash_check_enabled") is not True:
            reasons.append("image_hash_check_enabled is not true")
        if protocol.get("drop_empty_images") is not False:
            reasons.append("drop_empty_images is not false")
        if protocol.get("crowd_policy") != EXPECTED_CROWD_POLICY:
            reasons.append("unexpected crowd_policy")
        if protocol.get("category_policy") != EXPECTED_CATEGORY_POLICY:
            reasons.append("unexpected category_policy")
        partition = protocol.get("partition")
        alpha = protocol.get("dirichlet_alpha")
        if partition == "iid":
            if alpha is not None:
                reasons.append("IID dirichlet_alpha must be null")
            if protocol.get("partition_algorithm") != EXPECTED_IID_PARTITION_ALGORITHM:
                reasons.append("unexpected IID partition_algorithm")
        elif partition == "dirichlet":
            try:
                valid_alpha = math.isfinite(float(alpha)) and float(alpha) > 0
            except (TypeError, ValueError):
                valid_alpha = False
            if not valid_alpha:
                reasons.append("Dirichlet alpha must be finite and positive")
            if (
                protocol.get("partition_algorithm")
                != EXPECTED_DIRICHLET_PARTITION_ALGORITHM
            ):
                reasons.append("unexpected Dirichlet partition_algorithm")
        else:
            reasons.append("partition must be iid or dirichlet")
        try:
            valid_num_clients = int(protocol.get("num_clients")) >= 2
        except (TypeError, ValueError):
            valid_num_clients = False
        if not valid_num_clients:
            reasons.append("num_clients must be >=2")

        raw_counts = protocol.get("raw_source_counts")
        decode_check = protocol.get("image_decode_check")
        try:
            raw_image_count = sum(
                int(raw_counts[split]["images"])
                for split in ("train", "val", "test")
            )
            raw_count_valid = (
                isinstance(raw_counts, dict)
                and set(raw_counts) == {"train", "val", "test"}
                and raw_image_count > 0
            )
        except (KeyError, TypeError, ValueError):
            raw_count_valid = False
            raw_image_count = -1
        if not raw_count_valid:
            reasons.append("invalid raw_source_counts")
        if not (
            isinstance(decode_check, dict)
            and decode_check.get("enabled") is True
            and decode_check.get("method")
            == "pillow_verify_then_full_pixel_load"
            and decode_check.get("dimension_mismatches") == 0
            and decode_check.get("images_checked") == raw_image_count
            and decode_check.get("full_pixel_decodes") == raw_image_count
        ):
            reasons.append("incomplete full-pixel image decode audit")
        if not isinstance(summary, dict):
            reasons.append("missing source_hash_inventory_summary")
        else:
            if summary.get("schema_version") != 1:
                reasons.append("source inventory schema_version!=1")
            if summary.get("identity_policy") != EXPECTED_SOURCE_IDENTITY_POLICY:
                reasons.append("source inventory identity_policy mismatch")
            if not _is_sha256(summary.get("inventory_sha256")):
                reasons.append("invalid source inventory SHA-256")
            if not (
                isinstance(tree_hashes, dict)
                and set(tree_hashes) == {"train", "val", "test"}
                and all(_is_sha256(tree_hashes[split]) for split in tree_hashes)
            ):
                reasons.append("invalid per-split image-tree SHA-256 commitments")
        if not isinstance(post_policy, dict):
            reasons.append("invalid post-policy source check")
        else:
            duplicate_fields = (
                "filename_cross_split_duplicates",
                "roboflow_source_key_cross_split_duplicates",
                "sha256_cross_split_duplicates",
                "source_group_cross_split_duplicates",
            )
            if (
                post_policy.get("policy") != source_split_policy
                or post_policy.get("identity_policy")
                != EXPECTED_SOURCE_IDENTITY_POLICY
            ):
                reasons.append("post-policy source check metadata mismatch")
            if (
                source_split_policy == EXCLUSIVE_SOURCE_SPLIT_POLICY
                and any(post_policy.get(key) != 0 for key in duplicate_fields)
            ):
                reasons.append("source-exclusive post-policy check is not disjoint")
            if isinstance(summary, dict) and (
                post_policy.get("source_hash_inventory_sha256")
                != summary.get("inventory_sha256")
            ):
                reasons.append("post-policy/source-inventory SHA-256 mismatch")
        if not isinstance(source_audit, dict):
            reasons.append("invalid cross-split source audit")
        else:
            before_audit = source_audit.get("before")
            after_audit = source_audit.get("after")
            raw_signals = source_audit.get("raw_collision_signals")
            expected_after_overlap = (
                before_audit.get("cross_split_source_groups")
                if (
                    source_split_policy == OFFICIAL_SOURCE_SPLIT_POLICY
                    and isinstance(before_audit, dict)
                ) else 0
            )
            if (
                source_audit.get("policy") != source_split_policy
                or source_audit.get("identity_policy")
                != EXPECTED_SOURCE_IDENTITY_POLICY
                or not isinstance(after_audit, dict)
                or after_audit.get("cross_split_source_groups")
                != expected_after_overlap
            ):
                reasons.append("cross-split source audit is inconsistent")
            if source_audit.get("per_split_image_tree_sha256") != tree_hashes:
                reasons.append("source-audit/image-tree SHA-256 mismatch")
            if source_split_policy == OFFICIAL_SOURCE_SPLIT_POLICY:
                count_gate = protocol.get("official_count_gate")
                expected_counts = {
                    "train": {
                        "images": 15761, "annotations": 22058,
                        "background_images": 415,
                        "class_annotations": {
                            "airplane": 5508, "bird": 5522,
                            "drone": 5500, "helicopter": 5528,
                        },
                    },
                    "val": {
                        "images": 4514, "annotations": 6369,
                        "background_images": 125,
                        "class_annotations": {
                            "airplane": 1625, "bird": 1557,
                            "drone": 1602, "helicopter": 1585,
                        },
                    },
                    "test": {
                        "images": 2241, "annotations": 3171,
                        "background_images": 56,
                        "class_annotations": {
                            "airplane": 767, "bird": 821,
                            "drone": 796, "helicopter": 787,
                        },
                    },
                }
                if not (
                    isinstance(count_gate, dict)
                    and count_gate.get("enabled") is True
                    and count_gate.get("passed") is True
                    and count_gate.get("expected") == expected_counts
                    and count_gate.get("actual") == expected_counts
                ):
                    reasons.append("official AOD-4 v6 count gate is incomplete")
                if source_audit.get("excluded", {}).get("images") != 0:
                    reasons.append("official policy excluded images")
                before_per_split = source_audit.get("before", {}).get("per_split")
                after_per_split = source_audit.get("after", {}).get("per_split")
                if before_per_split != after_per_split:
                    reasons.append("official split membership was not preserved")
                source_counts = protocol.get("source_split_counts")
                selected_counts = protocol.get("split_counts")
                if source_counts != selected_counts:
                    reasons.append("official split counts differ from source counts")
                expected_duplicates = (
                    {
                        "filename_cross_split_duplicates": raw_signals.get(
                            "exact_filename_cross_split_keys"
                        ),
                        "roboflow_source_key_cross_split_duplicates": raw_signals.get(
                            "roboflow_source_key_cross_split_keys"
                        ),
                        "sha256_cross_split_duplicates": raw_signals.get(
                            "exact_sha256_cross_split_hashes"
                        ),
                        "source_group_cross_split_duplicates": before_audit.get(
                            "cross_split_source_groups"
                        ),
                    }
                    if isinstance(raw_signals, dict)
                    and isinstance(before_audit, dict)
                    else None
                )
                if (
                    not isinstance(post_policy, dict)
                    or expected_duplicates is None
                    or any(
                        post_policy.get(key) != value
                        for key, value in expected_duplicates.items()
                    )
                ):
                    reasons.append("official overlap counts are inconsistent")
        if reasons:
            invalid_schema7_protocol.append({
                "path": row["path"],
                "reasons": reasons,
            })
    if invalid_schema7_protocol:
        raise ValueError(
            "Publication aggregation accepts only complete schema-v7 "
            "AOD-4 source-policy manifests:\n"
            + json.dumps(invalid_schema7_protocol[:10], indent=2, ensure_ascii=False)
        )

    invalid = [
        row for row in rows
        if not _is_sha256(row.get("split_manifest_sha256"))
    ]
    if invalid:
        preview = ", ".join(
            f"{row['path']}={row.get('split_manifest_sha256')!r}"
            for row in invalid[:10]
        )
        raise ValueError(
            "Every publication result must record a valid immutable split SHA-256: "
            + preview
        )

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["seed"], row["partition_seed"], row["split_generation_signature"]
        )
        groups[key].append(row)
    mismatches = []
    for (seed, partition_seed, generation_signature), group in groups.items():
        digests = {row["split_manifest_sha256"] for row in group}
        if len(digests) > 1:
            mismatches.append({
                "seed": seed,
                "partition_seed": partition_seed,
                "split_generation_signature": generation_signature,
                "results": [
                    {
                        "method": row["method"],
                        "digest": row["split_manifest_sha256"],
                        "path": row["path"],
                    }
                    for row in group
                ],
            })
    if mismatches:
        raise ValueError(
            "Methods in the same paired replicate used different immutable splits:\n"
            + json.dumps(mismatches[:5], indent=2, ensure_ascii=False)
        )


def _collapse_local_clients(rows: list[dict]) -> list[dict]:
    """Make one Local observation per paired seed before computing run uncertainty."""
    grouped = defaultdict(list)
    nonlocal_rows = []
    for row in rows:
        if row["mode"] == "solo" or row["client_id"] is not None:
            grouped[(_group_key(row), row["seed"], row["partition_seed"])].append(row)
        else:
            nonlocal_rows.append(row)

    collapsed = list(nonlocal_rows)
    for (_, seed, partition_seed), clients in grouped.items():
        expected = clients[0]["num_clients"]
        digests = {row.get("split_manifest_sha256") for row in clients}
        if len(digests) != 1 or None in digests:
            raise ValueError(
                "Local clients in one seed pair must use one immutable split digest: "
                f"seed pair=({seed}, {partition_seed}), digests={sorted(map(str, digests))}"
            )
        if any(row["client_id"] is None for row in clients):
            raise ValueError(
                f"Local result is missing client_id for seed pair "
                f"({seed}, {partition_seed})"
            )
        client_ids = [int(row["client_id"]) for row in clients]
        duplicate_ids = sorted({cid for cid in client_ids if client_ids.count(cid) > 1})
        if duplicate_ids:
            duplicate_paths = {
                cid: [row["path"] for row in clients if int(row["client_id"]) == cid]
                for cid in duplicate_ids
            }
            raise ValueError(
                "Duplicate Local result(s) for the same client and seed pair "
                f"({seed}, {partition_seed}): {duplicate_paths}"
            )
        if len(clients) != expected:
            raise ValueError(
                f"Incomplete Local client count for seed pair ({seed}, {partition_seed}): "
                f"got {len(clients)} files, expected {expected}"
            )
        ids = set(client_ids)
        if ids != set(range(expected)):
            raise ValueError(
                f"Incomplete Local client set for seed pair ({seed}, {partition_seed}): "
                f"got {sorted(ids)}, "
                f"expected 0..{expected - 1}"
            )
        base = dict(clients[0])
        base["client_id"] = None
        base["path"] = ";".join(row["path"] for row in clients)
        for metric in ("AP", "AP50", "AP75"):
            values = np.asarray([row[f"macro_{metric}"] for row in clients], dtype=float)
            base[f"macro_{metric}"] = float(values.mean())
            base[f"client_sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            base[f"worst_{metric}"] = float(values.min())
            common_values = [
                value for value in (
                    _number(row[f"common_{metric}"]) for row in clients
                ) if math.isfinite(value)
            ]
            base[f"common_{metric}"] = (
                float(np.mean(common_values)) if common_values else float("nan")
            )
        for metric in ("mia_auc", "mia_tpr_at_1fpr", "mia_asr"):
            finite = [
                value for value in (_number(row[metric]) for row in clients)
                if math.isfinite(value)
            ]
            base[metric] = float(np.mean(finite)) if finite else float("nan")
        collapsed.append(base)
    return collapsed


def _validate_unique_paired_seeds(rows: list[dict], allow_duplicates: bool):
    occurrences = defaultdict(list)
    for row in rows:
        occurrences[(_group_key(row), row["seed"], row["partition_seed"])].append(row["path"])
    duplicates = {key: paths for key, paths in occurrences.items() if len(paths) > 1}
    if duplicates and not allow_duplicates:
        preview = "\n".join(f"{key}: {paths}" for key, paths in list(duplicates.items())[:10])
        raise ValueError(
            "Duplicate results for the same method/config/paired seed. Remove stale outputs or "
            f"pass --allow_duplicate_seeds explicitly:\n{preview}"
        )


def _mean_sd(values):
    finite = [
        number for number in (_number(value) for value in values)
        if math.isfinite(number)
    ]
    array = np.asarray(finite, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), 0
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        len(array),
    )


def summarize_runs(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {
            "method": key[0], "fl_method": key[1], "partition": key[2],
            "dirichlet_alpha": key[3], "lora_rank": key[4], "lora_alpha": key[5],
            "apply_lora_backbone": key[6], "apply_lora_decoder": key[7],
            "num_clients": key[8],
            "paired_seed_count": len({
                (row["seed"], row["partition_seed"]) for row in group
            }),
            "protocol_signature": key[9],
            "protocol_manifest": group[0]["protocol_manifest"],
            "seeds": sorted({row["seed"] for row in group}),
            "partition_seeds": sorted({row["partition_seed"] for row in group}),
            "seed_pairs": [
                {"seed": seed, "partition_seed": partition_seed}
                for seed, partition_seed in sorted({
                    (row["seed"], row["partition_seed"]) for row in group
                })
            ],
            "split_digests_by_seed_pair": [
                {
                    "seed": seed,
                    "partition_seed": partition_seed,
                    "split_manifest_sha256": next(
                        row["split_manifest_sha256"] for row in group
                        if row["seed"] == seed
                        and row["partition_seed"] == partition_seed
                    ),
                }
                for seed, partition_seed in sorted({
                    (row["seed"], row["partition_seed"]) for row in group
                })
            ],
        }
        for field in METRICS:
            mean, sd, count = _mean_sd([row[field] for row in group])
            summary[f"{field}_mean"] = mean
            summary[f"{field}_run_sd"] = sd
            summary[f"{field}_n"] = count
        for field in (
            "total_params", "trainable_params", "communication_params", "trainable_ratio_pct",
            "parameter_saving_pct", "communication_saving_pct",
            "communication_byte_saving_pct", "communication_element_saving_pct",
            "one_client_one_way_mb",
            "system_round_total_mb", "cumulative_total_mb",
            "total_params_m", "trainable_params_m", "communication_params_m",
        ):
            summary[field] = _mean_sd([row[field] for row in group])[0]
        summaries.append(summary)
    return summaries


def _comparison_family_signature(row: dict) -> str:
    protocol = {
        key: value for key, value in row["protocol_manifest"].items()
        if key not in METHOD_SPECIFIC_PROTOCOL_KEYS
    }
    serialized = json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _paired_bootstrap_interval(deltas: np.ndarray, seed: int, resamples: int = 20_000):
    if deltas.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = deltas[rng.integers(0, deltas.size, size=(resamples, deltas.size))]
    means = sampled.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_fedsa_comparisons(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pair FedSA with FL baselines at identical train/partition seeds and protocol."""
    federated = [row for row in rows if row["mode"] == "fl"]
    baseline_index = defaultdict(list)
    for row in federated:
        if row["fl_method"] not in ("full_ft", "lora"):
            continue
        key = (
            _comparison_family_signature(row), row["seed"], row["partition_seed"],
            row["partition"], row["dirichlet_alpha"], row["num_clients"], row["fl_method"],
        )
        baseline_index[key].append(row)

    paired_rows = []
    for proposal in federated:
        if proposal["fl_method"] != "fedsa_lora":
            continue
        # The role-splitting contribution is compared on the declared both-target
        # primary model. Target-only proposal ablations remain separate controls.
        if not (
            proposal.get("apply_lora_backbone") is True
            and proposal.get("apply_lora_decoder") is True
        ):
            continue
        family = _comparison_family_signature(proposal)
        for baseline_method in ("full_ft", "lora"):
            key = (
                family, proposal["seed"], proposal["partition_seed"],
                proposal["partition"], proposal["dirichlet_alpha"],
                proposal["num_clients"], baseline_method,
            )
            candidates = baseline_index.get(key, [])
            if baseline_method == "lora":
                candidates = [
                    row for row in candidates
                    if row["lora_rank"] == proposal["lora_rank"]
                    and row["lora_alpha"] == proposal["lora_alpha"]
                    and row.get("apply_lora_backbone") is True
                    and row.get("apply_lora_decoder") is True
                ]
            if len(candidates) > 1:
                raise ValueError(
                    "Ambiguous paired baseline for FedSA comparison: "
                    f"seed={proposal['seed']}, baseline={baseline_method}"
                )
            if not candidates:
                continue
            baseline = candidates[0]
            if proposal["split_manifest_sha256"] != baseline["split_manifest_sha256"]:
                raise ValueError(
                    "Paired FedSA and baseline results used different immutable splits: "
                    f"seed pair=({proposal['seed']}, {proposal['partition_seed']}), "
                    f"proposal={proposal['split_manifest_sha256']}, "
                    f"baseline={baseline['split_manifest_sha256']}"
                )
            record = {
                "proposal": proposal["method"],
                "baseline": baseline["method"],
                "partition": proposal["partition"],
                "dirichlet_alpha": proposal["dirichlet_alpha"],
                "lora_rank": proposal["lora_rank"],
                "lora_alpha": proposal["lora_alpha"],
                "apply_lora_backbone": proposal["apply_lora_backbone"],
                "apply_lora_decoder": proposal["apply_lora_decoder"],
                "num_clients": proposal["num_clients"],
                "protocol_family_signature": family,
                "seed": proposal["seed"],
                "partition_seed": proposal["partition_seed"],
                "split_manifest_sha256": proposal["split_manifest_sha256"],
                "proposal_path": proposal["path"],
                "baseline_path": baseline["path"],
            }
            for metric in PAIRED_METRICS:
                proposal_value = float(proposal[metric])
                baseline_value = float(baseline[metric])
                record[f"proposal_{metric}"] = proposal_value
                record[f"baseline_{metric}"] = baseline_value
                record[f"delta_{metric}"] = proposal_value - baseline_value
            paired_rows.append(record)

    grouped = defaultdict(list)
    for row in paired_rows:
        key = (
            row["proposal"], row["baseline"], row["partition"], row["dirichlet_alpha"],
            row["lora_rank"], row["lora_alpha"], row["apply_lora_backbone"],
            row["apply_lora_decoder"], row["num_clients"],
            row["protocol_family_signature"],
        )
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        seed_pairs = sorted({
            (int(row["seed"]), int(row["partition_seed"])) for row in group
        })
        if len(seed_pairs) != len(group):
            raise ValueError(f"Duplicate seed pair in paired comparison: {key}")
        seeds = sorted({pair[0] for pair in seed_pairs})
        partition_seeds = sorted({pair[1] for pair in seed_pairs})
        bootstrap_seed_material = json.dumps(seed_pairs, separators=(",", ":"))
        bootstrap_seed_base = int(hashlib.sha256(
            bootstrap_seed_material.encode("utf-8")
        ).hexdigest()[:8], 16)
        for metric_index, metric in enumerate(PAIRED_METRICS):
            deltas = np.asarray([row[f"delta_{metric}"] for row in group], dtype=np.float64)
            if not np.all(np.isfinite(deltas)):
                continue
            lower, upper = _paired_bootstrap_interval(
                deltas,
                seed=1729 + metric_index + bootstrap_seed_base,
            )
            mean_delta = float(deltas.mean())
            improvement = -mean_delta if metric in LOWER_IS_BETTER else mean_delta
            summaries.append({
                "proposal": key[0], "baseline": key[1], "partition": key[2],
                "dirichlet_alpha": key[3], "lora_rank": key[4], "lora_alpha": key[5],
                "apply_lora_backbone": key[6], "apply_lora_decoder": key[7],
                "num_clients": key[8], "protocol_family_signature": key[9],
                "metric": metric,
                "preferred_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                "paired_seed_count": len(deltas), "seeds": seeds,
                "partition_seeds": partition_seeds,
                "seed_pairs": [
                    {"seed": seed, "partition_seed": partition_seed}
                    for seed, partition_seed in seed_pairs
                ],
                "split_digests_by_seed_pair": [
                    {
                        "seed": row["seed"],
                        "partition_seed": row["partition_seed"],
                        "split_manifest_sha256": row["split_manifest_sha256"],
                    }
                    for row in sorted(
                        group, key=lambda item: (item["seed"], item["partition_seed"])
                    )
                ],
                "mean_delta_proposal_minus_baseline": mean_delta,
                "sample_sd_of_paired_deltas": (
                    float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                ),
                "mean_improvement_in_preferred_direction": float(improvement),
                "paired_bootstrap_95_ci_lower": lower,
                "paired_bootstrap_95_ci_upper": upper,
                "wins": int(np.sum(deltas < 0 if metric in LOWER_IS_BETTER else deltas > 0)),
            })
    return paired_rows, summaries


def _summarize_generic_pairs(
    paired_rows: list[dict],
    metrics: tuple[str, ...],
    lower_is_better: set[str],
    *,
    bootstrap_seed_offset: int,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in paired_rows:
        key = (
            row["proposal"], row["baseline"], row["partition"],
            row["dirichlet_alpha"], row["lora_rank"], row["lora_alpha"],
            row["apply_lora_backbone"], row["apply_lora_decoder"],
            row["num_clients"], row["protocol_family_signature"],
        )
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        seed_pairs = sorted({
            (int(row["seed"]), int(row["partition_seed"])) for row in group
        })
        if len(seed_pairs) != len(group):
            raise ValueError(f"Duplicate seed pair in paired comparison: {key}")
        bootstrap_seed_base = int(hashlib.sha256(
            json.dumps(seed_pairs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8], 16)
        for metric_index, metric in enumerate(metrics):
            deltas = np.asarray(
                [row[f"delta_{metric}"] for row in group], dtype=np.float64
            )
            if not np.all(np.isfinite(deltas)):
                continue
            lower, upper = _paired_bootstrap_interval(
                deltas,
                seed=bootstrap_seed_offset + metric_index + bootstrap_seed_base,
            )
            mean_delta = float(deltas.mean())
            prefer_lower = metric in lower_is_better
            summaries.append({
                "proposal": key[0], "baseline": key[1], "partition": key[2],
                "dirichlet_alpha": key[3], "lora_rank": key[4],
                "lora_alpha": key[5], "apply_lora_backbone": key[6],
                "apply_lora_decoder": key[7], "num_clients": key[8],
                "protocol_family_signature": key[9], "metric": metric,
                "preferred_direction": "lower" if prefer_lower else "higher",
                "paired_seed_count": len(deltas),
                "seeds": sorted({pair[0] for pair in seed_pairs}),
                "partition_seeds": sorted({pair[1] for pair in seed_pairs}),
                "seed_pairs": [
                    {"seed": seed, "partition_seed": partition_seed}
                    for seed, partition_seed in seed_pairs
                ],
                "split_digests_by_seed_pair": [
                    {
                        "seed": row["seed"],
                        "partition_seed": row["partition_seed"],
                        "split_manifest_sha256": row["split_manifest_sha256"],
                    }
                    for row in sorted(
                        group, key=lambda item: (item["seed"], item["partition_seed"])
                    )
                ],
                "mean_delta_proposal_minus_baseline": mean_delta,
                "sample_sd_of_paired_deltas": (
                    float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                ),
                "mean_improvement_in_preferred_direction": (
                    -mean_delta if prefer_lower else mean_delta
                ),
                "paired_bootstrap_95_ci_lower": lower,
                "paired_bootstrap_95_ci_upper": upper,
                "wins": int(np.sum(deltas < 0 if prefer_lower else deltas > 0)),
            })
    return summaries


def paired_factor_sharing_comparisons(
    rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pair global-A/local-B FedSA against global-B/local-A at one protocol."""
    federated = [row for row in rows if row["mode"] == "fl"]
    fixed_b_index = defaultdict(list)
    for row in federated:
        if row["fl_method"] != "fixed_share_b_lora":
            continue
        key = (
            _comparison_family_signature(row), row["seed"], row["partition_seed"],
            row["partition"], row["dirichlet_alpha"], row["num_clients"],
            row["lora_rank"], row["lora_alpha"],
            row.get("apply_lora_backbone"), row.get("apply_lora_decoder"),
        )
        fixed_b_index[key].append(row)

    paired_rows = []
    for proposal in federated:
        if proposal["fl_method"] != "fedsa_lora":
            continue
        if not (
            proposal.get("apply_lora_backbone") is True
            and proposal.get("apply_lora_decoder") is True
        ):
            continue
        family = _comparison_family_signature(proposal)
        key = (
            family, proposal["seed"], proposal["partition_seed"],
            proposal["partition"], proposal["dirichlet_alpha"],
            proposal["num_clients"], proposal["lora_rank"],
            proposal["lora_alpha"], proposal.get("apply_lora_backbone"),
            proposal.get("apply_lora_decoder"),
        )
        candidates = fixed_b_index.get(key, [])
        if len(candidates) > 1:
            raise ValueError(
                "Ambiguous Fixed Share-B result for factor-sharing comparison: "
                f"seed pair=({proposal['seed']}, {proposal['partition_seed']})"
            )
        if not candidates:
            continue
        baseline = candidates[0]
        if proposal["split_manifest_sha256"] != baseline["split_manifest_sha256"]:
            raise ValueError(
                "Paired factor-sharing results used different immutable splits: "
                f"seed pair=({proposal['seed']}, {proposal['partition_seed']})"
            )
        differing_controls = [
            field for field in METHOD_SPECIFIC_PROTOCOL_KEYS - {"fl_method"}
            if proposal["protocol_manifest"].get(field)
            != baseline["protocol_manifest"].get(field)
        ]
        if differing_controls:
            raise ValueError(
                "FedSA and Fixed Share-B must differ only in factor-sharing role; "
                f"mismatched controls={sorted(differing_controls)}"
            )
        record = {
            "proposal": proposal["method"],
            "baseline": baseline["method"],
            "partition": proposal["partition"],
            "dirichlet_alpha": proposal["dirichlet_alpha"],
            "lora_rank": proposal["lora_rank"],
            "lora_alpha": proposal["lora_alpha"],
            "apply_lora_backbone": proposal["apply_lora_backbone"],
            "apply_lora_decoder": proposal["apply_lora_decoder"],
            "num_clients": proposal["num_clients"],
            "protocol_family_signature": family,
            "seed": proposal["seed"],
            "partition_seed": proposal["partition_seed"],
            "split_manifest_sha256": proposal["split_manifest_sha256"],
            "proposal_path": proposal["path"],
            "baseline_path": baseline["path"],
            "delta_direction": "FedSA(global_A_local_B)-FixedShareB(global_B_local_A)",
        }
        for metric in FACTOR_SHARING_PAIRED_METRICS:
            proposal_value = float(proposal[metric])
            baseline_value = float(baseline[metric])
            record[f"proposal_{metric}"] = proposal_value
            record[f"baseline_{metric}"] = baseline_value
            record[f"delta_{metric}"] = proposal_value - baseline_value
        paired_rows.append(record)

    return paired_rows, _summarize_generic_pairs(
        paired_rows,
        FACTOR_SHARING_PAIRED_METRICS,
        FACTOR_SHARING_LOWER_IS_BETTER,
        bootstrap_seed_offset=2718,
    )


def _write_csv(rows: list[dict], path: str):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _format(value, digits=4):
    value = _number(value)
    return "N/A" if not math.isfinite(value) else f"{value:.{digits}f}"


def _target_label(item: dict) -> str:
    backbone = item.get("apply_lora_backbone")
    decoder = item.get("apply_lora_decoder")
    if backbone is None and decoder is None:
        return ""
    if backbone and decoder:
        return "both-targets"
    if backbone:
        return "backbone-only"
    if decoder:
        return "decoder-only"
    return "no-targets"


def _config_label(item: dict, compact: bool = False) -> str:
    separator = " " if compact else " / "
    label = str(item["method"])
    if item["partition"] == "dirichlet":
        label += f"{separator}a={item['dirichlet_alpha']}"
    else:
        label += f"{separator}IID"
    if item["lora_rank"] is not None:
        label += f"{separator}r={item['lora_rank']}"
    target = _target_label(item)
    if target:
        label += f"{separator}{target}"
    return label


def _write_markdown(summaries: list[dict], path: str):
    lines = [
        "# Experiment summary",
        "",
        "Values are paired-run mean ± run sample SD. `Client SD` is the within-run client "
        "sample SD and is not a confidence interval.",
        "A replicate is `(training seed, partition seed)`. Its methods share one validated "
        "split SHA-256; different partition-seed replicates may use different immutable splits.",
        "",
        "| Method/config | Runs | Macro AP | AP50 | AP75 | Client SD AP/AP50/AP75 | Worst AP | Common AP | MIA AUC | Total comm. MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_configs = [
        _config_label(item).replace("a=", "alpha=") for item in summaries
    ]
    config_counts = {
        config: base_configs.count(config) for config in set(base_configs)
    }
    for item, base_config in zip(summaries, base_configs):
        config = base_config
        if config_counts[base_config] > 1:
            config += f" / protocol={item['protocol_signature'][:8]}"
        macro_ap = f"{_format(item['macro_AP_mean'])} ± {_format(item['macro_AP_run_sd'])}"
        ap50 = f"{_format(item['macro_AP50_mean'])} ± {_format(item['macro_AP50_run_sd'])}"
        ap75 = f"{_format(item['macro_AP75_mean'])} ± {_format(item['macro_AP75_run_sd'])}"
        lines.append(
            "| " + " | ".join([
                config,
                str(item["paired_seed_count"]),
                macro_ap,
                ap50,
                ap75,
                "/".join(_format(item[f"client_sd_{metric}_mean"]) for metric in ("AP", "AP50", "AP75")),
                _format(item["worst_AP_mean"]),
                _format(item["common_AP_mean"]),
                _format(item["mia_auc_mean"]),
                _format(item["cumulative_total_mb"], 2),
            ]) + " |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_paired_markdown(
    comparisons: list[dict], path: str, *,
    title: str = "Paired FedSA-LoRA comparisons",
    selected_metrics: Optional[set[str]] = None,
):
    lines = [
        f"# {title}",
        "",
        "Each delta is proposal minus baseline at the same training seed and partition seed. "
        "Intervals are paired non-parametric bootstrap percentile intervals. With only three "
        "seeds they are exploratory uncertainty summaries, not confirmatory significance tests.",
        "The split SHA-256 is required to match within each pair and is allowed to differ "
        "between partition-seed replicates generated by the same recorded protocol.",
        "",
        "| Proposal vs baseline | Setting | Metric | Pairs | Mean delta | 95% interval | Wins |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    selected = selected_metrics or {
        "macro_AP", "client_sd_AP", "worst_AP", "common_AP"
    }
    for row in comparisons:
        if row["metric"] not in selected:
            continue
        setting = (
            f"alpha={row['dirichlet_alpha']}"
            if row["partition"] == "dirichlet"
            else "IID"
        )
        setting += f", r={row['lora_rank']}"
        setting += f", protocol={row['protocol_family_signature'][:8]}"
        interval = (
            f"[{_format(row['paired_bootstrap_95_ci_lower'])}, "
            f"{_format(row['paired_bootstrap_95_ci_upper'])}]"
        )
        lines.append(
            "| " + " | ".join([
                f"{row['proposal']} vs {row['baseline']}", setting, row["metric"],
                str(row["paired_seed_count"]),
                _format(row["mean_delta_proposal_minus_baseline"]), interval,
                str(row["wins"]),
            ]) + " |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _validate_result_envelope(path: str, result) -> None:
    """Apply the same completion gate used by experiment skip/resume scripts."""
    if not isinstance(result, dict):
        raise ValueError(f"Result root is not an object: {path}")
    if result.get("status") != "complete":
        raise ValueError(f"Result has no status='complete' marker: {path}")
    schema_version = _integer(result.get("result_schema_version"), -1)
    if schema_version < 2:
        raise ValueError(
            f"Unsupported or missing result_schema_version in {path}: "
            f"{result.get('result_schema_version')!r}"
        )
    mia = result.get("mia")
    if mia is not None:
        expected_policy = (
            "exclude_test_source_components_present_in_any_train_client"
        )
        configuration = mia.get("configuration") if isinstance(mia, dict) else None
        if not isinstance(configuration, dict) or configuration.get(
            "nonmember_source_policy"
        ) != expected_policy:
            raise ValueError(
                f"MIA result lacks the required source-disjoint nonmember policy: {path}"
            )
        records = mia.get("per_client")
        if not isinstance(records, list) or not records:
            raise ValueError(f"MIA result has no per-client records: {path}")
        for index, entry in enumerate(records):
            metrics = entry.get("metrics", entry) if isinstance(entry, dict) else None
            source_audit = (
                metrics.get("source_disjoint_sampling")
                if isinstance(metrics, dict) else None
            )
            if not (
                isinstance(source_audit, dict)
                and source_audit.get("policy") == expected_policy
                and source_audit.get(
                    "member_nonmember_source_group_intersection"
                ) == 0
            ):
                raise ValueError(
                    f"MIA client {index} is not source-disjoint in {path}"
                )


def _plot_summaries(summaries: list[dict], output_dir: str):
    from utils.visualization import (
        plot_communication_cost,
        plot_mia_results,
        plot_rank_sensitivity,
    )

    base_labels = [_config_label(item, compact=True) for item in summaries]
    label_counts = {label: base_labels.count(label) for label in set(base_labels)}
    labels = {
        index: (
            label
            if label_counts[label] == 1
            else f"{label} p={item['protocol_signature'][:6]}"
        )
        for index, (item, label) in enumerate(zip(summaries, base_labels))
    }

    communication = {}
    mia = {}
    rank = {}
    for index, item in enumerate(summaries):
        label = labels[index]
        if math.isfinite(_number(item["cumulative_total_mb"])):
            communication[label] = {
                "total_comm_mb": item["cumulative_total_mb"],
                "trainable_params_m": item["trainable_params"] / 1e6,
                "comm_params_m": item["communication_params"] / 1e6,
            }
        if math.isfinite(_number(item["mia_auc_mean"])):
            mia[label] = {
                "auc_roc": item["mia_auc_mean"],
                "tpr_at_1fpr": item["mia_tpr_at_1fpr_mean"],
                "asr": item["mia_asr_mean"],
            }
        if (
            item["fl_method"] == "fedsa_lora"
            and item["partition"] == "dirichlet"
            and _number(item["dirichlet_alpha"]) == 0.4
            and item["lora_rank"] is not None
            and item.get("apply_lora_backbone") is True
            and item.get("apply_lora_decoder") is True
        ):
            rank[int(item["lora_rank"])] = {
                "AP": item["macro_AP_mean"], "AP50": item["macro_AP50_mean"],
                "trainable_params_m": item["trainable_params"] / 1e6,
            }
    if communication:
        plot_communication_cost(communication, os.path.join(output_dir, "communication_cost.png"))
    if mia:
        plot_mia_results(mia, os.path.join(output_dir, "mia_results.png"))
    if len(rank) >= 2:
        plot_rank_sensitivity(rank, os.path.join(output_dir, "rank_sensitivity.png"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="./results")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--allow_duplicate_seeds", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results_dir, "**", "*_results.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_results.json files under {args.results_dir}")
    raw_rows = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        _validate_result_envelope(path, result)
        raw_rows.append(normalize_result(path, result))

    _validate_split_digest_consistency(raw_rows)
    rows = _collapse_local_clients(raw_rows)
    _validate_unique_paired_seeds(rows, args.allow_duplicate_seeds)
    summaries = summarize_runs(rows)
    paired_rows, paired_summaries = paired_fedsa_comparisons(rows)
    factor_rows, factor_summaries = paired_factor_sharing_comparisons(rows)
    output_dir = os.path.abspath(args.output_dir or os.path.join(args.results_dir, "summary"))
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(raw_rows, os.path.join(output_dir, "individual_results.csv"))
    _write_csv(rows, os.path.join(output_dir, "paired_run_results.csv"))
    _write_csv(summaries, os.path.join(output_dir, "summary_by_method.csv"))
    _write_csv(paired_rows, os.path.join(output_dir, "paired_fedsa_seed_deltas.csv"))
    _write_csv(
        paired_summaries, os.path.join(output_dir, "paired_fedsa_comparisons.csv")
    )
    _write_csv(
        factor_rows,
        os.path.join(output_dir, "paired_factor_sharing_seed_deltas.csv"),
    )
    _write_csv(
        factor_summaries,
        os.path.join(output_dir, "paired_factor_sharing_comparisons.csv"),
    )
    _write_markdown(summaries, os.path.join(output_dir, "summary.md"))
    _write_paired_markdown(
        paired_summaries, os.path.join(output_dir, "paired_fedsa_comparisons.md")
    )
    _write_paired_markdown(
        factor_summaries,
        os.path.join(output_dir, "paired_factor_sharing_comparisons.md"),
        title="Paired fixed factor-sharing comparisons",
        selected_metrics={
            "macro_AP", "client_sd_AP", "worst_AP", "common_AP",
            "communication_params", "one_client_one_way_mb",
            "cumulative_total_mb",
        },
    )
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summaries), handle, indent=2, ensure_ascii=False, allow_nan=False)
    with open(
        os.path.join(output_dir, "paired_fedsa_comparisons.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(paired_summaries), handle, indent=2,
            ensure_ascii=False, allow_nan=False,
        )
    with open(
        os.path.join(output_dir, "paired_factor_sharing_comparisons.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(factor_summaries), handle, indent=2,
            ensure_ascii=False, allow_nan=False,
        )
    if not args.no_plots:
        _plot_summaries(summaries, output_dir)
    print(f"Aggregated {len(files)} result files into {output_dir}")


if __name__ == "__main__":
    main()
