#!/usr/bin/env python3
"""Fail-fast end-to-end preflight for AOD-4 RT-DETR experiments.

This command does not train or evaluate a checkpoint. It verifies the immutable
split manifest, exact YOLO tree, RT-DETR/LoRA injection policy, and one labeled
forward/backward batch before a long experiment is submitted.

Use the same arguments as ``main.py``::

    python3 scripts/preflight.py \
      --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
      --split_file data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json \
      --mode fl --fl_method fedsa_lora --device cuda
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import get_args
from data.dataset import (
    AOD4_CATEGORY_POLICY,
    AOD4_V6_OFFICIAL_COUNTS,
    CLIENT_PARTITION_UNIT,
    SOURCE_GROUP_FIELD,
    SOURCE_IDENTITY_POLICY,
    SOURCE_SPLIT_POLICY_EXCLUSIVE,
    SOURCE_SPLIT_POLICY_OFFICIAL,
    SOURCE_SPLIT_PRIORITY,
    SUPPORTED_SOURCE_SPLIT_POLICIES,
    SPLITS,
    _clip_bbox_xywh,
    align_cross_split_source_group_owners,
    apply_source_split_policy,
    annotation_sha256,
    build_image_ann_map,
    build_source_hash_inventory,
    canonicalize_aod4_categories,
    category_mapping,
    draw_client_proportions,
    filter_annotations,
    generated_yolo_tree_sha256,
    load_coco_annotations,
    partition_images,
    partition_images_iid,
    source_image_key,
    split_statistics,
    validate_coco,
    validate_source_group_assignments,
)
from models.lora import LoRAConv2d, LoRALinear, LoRAMultiheadAttention
from models.rtdetr_lora import RTDETRLoRA
from trainers.trainer import _move_batch, _unpack_loss, build_rtdetr_dataset


def _pass(message: str):
    print(f"[PASS] {message}")


def _require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _normalise_names(names) -> List[str]:
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        converted = {int(key): str(value) for key, value in names.items()}
        expected = list(range(len(converted)))
        if sorted(converted) != expected:
            raise ValueError(f"Dataset names keys must be contiguous 0..nc-1, got {sorted(converted)}")
        return [converted[index] for index in expected]
    raise TypeError(f"Unsupported dataset names value: {type(names).__name__}")


def _derive_yolo_dir(split_file: str) -> str:
    stem = os.path.splitext(os.path.basename(split_file))[0]
    if not stem.startswith("split_"):
        raise ValueError(f"Split manifest name must start with 'split_': {split_file}")
    return os.path.join(
        os.path.dirname(os.path.abspath(split_file)),
        stem.replace("split_", "yolo_", 1),
    )


def _client_records(manifest: dict, num_clients: int) -> List[dict]:
    clients = manifest.get("clients")
    if not isinstance(clients, list):
        raise ValueError("Split manifest has no list-valued 'clients'")
    clients = sorted(clients, key=lambda item: int(item["client_id"]))
    ids = [int(item["client_id"]) for item in clients]
    if ids != list(range(num_clients)):
        raise ValueError(f"Client IDs must be 0..{num_clients - 1}, got {ids}")
    return clients


def validate_manifest(args) -> Tuple[dict, Dict[str, dict], Dict[str, List[List[int]]]]:
    """Reproduce the complete schema-v7 source policy and partition pipeline."""
    manifest = _load_json(args.split_file)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Split manifest has no metadata mapping")
    required_metadata = {
        "schema_version", "data_root", "partition", "dirichlet_alpha",
        "partition_algorithm", "client_partition_unit", "num_clients", "seed",
        "min_bbox_area", "min_bbox_side", "drop_empty_images", "crowd_policy",
        "category_policy", "source_category_audit", "source_split_policy",
        "official_split_preserved",
        "source_identity_policy", "source_split_priority", "source_hash_inventory",
        "raw_source_counts", "source_split_counts", "split_counts", "annotation_sha256",
        "official_count_gate",
        "cross_split_source_audit", "image_hash_check_enabled",
        "post_policy_cross_split_source_check",
        "image_decode_check", "client_target_proportions",
        "cross_split_client_source_group_check",
        "realized_partition_statistics", "quantity_balance",
        "class_names", "cat_id_to_label", "generated_yolo_tree_sha256",
    }
    missing = required_metadata - set(metadata)
    if missing:
        raise ValueError(f"Split metadata is missing keys: {sorted(missing)}")
    if int(metadata["schema_version"]) != 7:
        raise ValueError(
            f"Unsupported split schema {metadata['schema_version']!r}; expected 7"
        )
    if os.path.abspath(metadata["data_root"]) != os.path.abspath(args.data_root):
        raise ValueError(
            f"Manifest data_root={metadata['data_root']!r} differs from CLI "
            f"{args.data_root!r}"
        )
    source_split_policy = metadata["source_split_policy"]
    if source_split_policy not in SUPPORTED_SOURCE_SPLIT_POLICIES:
        raise ValueError("Manifest source-split policy is stale or unsupported")
    if bool(metadata["official_split_preserved"]) != (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
    ):
        raise ValueError("Manifest official-preservation marker is inconsistent")
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and (
            float(metadata["min_bbox_area"]) != 0.0
            or float(metadata["min_bbox_side"]) != 0.0
        )
    ):
        raise ValueError("Official AOD-4 v6 manifest uses bbox filtering")
    if metadata["source_identity_policy"] != SOURCE_IDENTITY_POLICY:
        raise ValueError("Manifest source-identity policy is stale or unsupported")
    expected_priority = (
        [] if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        else list(SOURCE_SPLIT_PRIORITY)
    )
    if list(metadata["source_split_priority"]) != expected_priority:
        raise ValueError("Manifest source split priority is stale or unsupported")
    if metadata["client_partition_unit"] != CLIENT_PARTITION_UNIT:
        raise ValueError("Manifest client partition unit is not source-group atomic")
    expected_partition_algorithm = (
        "random_source_group_lpt_balance_cross_split_owner_v2"
        if args.partition == "iid"
        else "source_group_target_deficit_balance_cross_split_owner_v2"
    )
    if metadata["partition_algorithm"] != expected_partition_algorithm:
        raise ValueError("Manifest source-group partition algorithm is stale")
    if metadata["category_policy"] != AOD4_CATEGORY_POLICY:
        raise ValueError("Manifest AOD-4 category policy is stale or unsupported")
    if bool(metadata["drop_empty_images"]):
        raise ValueError("Primary AOD-4 manifest must retain background images")
    if metadata["crowd_policy"] != (
        "require_zero_crowd_annotations_for_YOLO_metric_equivalence"
    ):
        raise ValueError("Manifest crowd policy is incompatible with the protocol")
    if metadata["image_hash_check_enabled"] is not True:
        raise ValueError("Schema-v7 requires SHA-256 inventory for every raw image")

    checks = {
        "num_clients": (int(metadata["num_clients"]), int(args.num_clients)),
        "partition_seed": (int(metadata["seed"]), int(args.partition_seed)),
        "partition": (str(metadata["partition"]), str(args.partition)),
    }
    mismatches = [
        f"{key}: split={left!r}, CLI={right!r}"
        for key, (left, right) in checks.items() if left != right
    ]
    for name in ("min_bbox_area", "min_bbox_side"):
        if not math.isclose(
            float(metadata[name]), float(getattr(args, name)),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            mismatches.append(
                f"{name}: split={metadata[name]!r}, CLI={getattr(args, name)!r}"
            )
    if args.partition == "dirichlet" and not math.isclose(
        float(metadata["dirichlet_alpha"]), float(args.dirichlet_alpha),
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        mismatches.append(
            "dirichlet_alpha: "
            f"split={metadata['dirichlet_alpha']!r}, CLI={args.dirichlet_alpha!r}"
        )
    if mismatches:
        raise ValueError("Split/CLI configuration mismatch:\n  " + "\n  ".join(mismatches))

    stored_category_audit = metadata["source_category_audit"]
    if not isinstance(stored_category_audit, dict) or set(stored_category_audit) != set(SPLITS):
        raise ValueError("Manifest source_category_audit must cover train/val/test")
    stored_source_counts = metadata["source_split_counts"]
    if not isinstance(stored_source_counts, dict) or set(stored_source_counts) != set(SPLITS):
        raise ValueError("Manifest source_split_counts must cover train/val/test")
    stored_raw_source_counts = metadata["raw_source_counts"]
    if (
        not isinstance(stored_raw_source_counts, dict)
        or set(stored_raw_source_counts) != set(SPLITS)
    ):
        raise ValueError("Manifest raw_source_counts must cover train/val/test")
    stored_inventory = metadata["source_hash_inventory"]
    if not isinstance(stored_inventory, dict):
        raise ValueError("Manifest has no source hash inventory mapping")
    stored_source_audit = metadata["cross_split_source_audit"]
    if not isinstance(stored_source_audit, dict):
        raise ValueError("Manifest cross-split source audit is not a mapping")
    expected_after_overlap = (
        int(stored_source_audit.get("before", {}).get(
            "cross_split_source_groups", -1
        ))
        if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL else 0
    )
    if (
        stored_source_audit.get("policy") != source_split_policy
        or stored_source_audit.get("identity_policy") != SOURCE_IDENTITY_POLICY
        or list(stored_source_audit.get("split_priority", [])) != expected_priority
        or int(stored_source_audit.get("after", {}).get(
            "cross_split_source_groups", -1
        )) != expected_after_overlap
    ):
        raise ValueError("Manifest cross-split source audit is stale or inconsistent")

    raw_by_split = {}
    canonical_by_split = {}
    annotation_hashes = {}
    expected_declared_categories = None
    expected_target_categories = None
    actual_official_counts = {}
    for split in SPLITS:
        image_dir = os.path.join(args.data_root, split)
        annotation_path = os.path.join(image_dir, "_annotations.coco.json")
        current_annotation_hash = annotation_sha256(annotation_path)
        if current_annotation_hash != metadata["annotation_sha256"][split]:
            raise ValueError(f"{split}: annotation JSON hash differs from manifest")
        annotation_hashes[split] = current_annotation_hash
        raw = load_coco_annotations(annotation_path)
        raw_by_split[split] = raw
        stats = validate_coco(
            raw, image_dir, split, expected_categories=expected_declared_categories
        )
        actual_official_counts[split] = {
            "images": int(stats["images"]),
            "annotations": int(stats["annotations"]),
            "background_images": int(stats["background_images"]),
        }
        if int(stats["crowd_annotations"]) != 0:
            raise ValueError(
                f"{split}: crowd annotations cannot be represented by YOLO labels"
            )
        current_declared = category_mapping(raw)
        if expected_declared_categories is None:
            expected_declared_categories = current_declared
        canonical, category_audit = canonicalize_aod4_categories(raw)
        if category_audit != stored_category_audit[split]:
            raise ValueError(
                f"{split}: source category audit differs from the immutable manifest"
            )
        current_targets = category_mapping(canonical)
        if expected_target_categories is None:
            expected_target_categories = current_targets
        elif current_targets != expected_target_categories:
            raise ValueError(
                f"{split}: canonical AOD-4 target mapping differs from train"
            )
        actual_official_counts[split]["class_annotations"] = {
            current_targets[category_id]: sum(
                int(annotation["category_id"]) == category_id
                for annotation in canonical["annotations"]
            )
            for category_id in sorted(current_targets)
        }
        canonical_by_split[split] = canonical

    official_count_gate = metadata["official_count_gate"]
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and not (
            isinstance(official_count_gate, dict)
            and official_count_gate.get("enabled") is True
            and official_count_gate.get("passed") is True
            and official_count_gate.get("expected") == AOD4_V6_OFFICIAL_COUNTS
            and official_count_gate.get("actual") == actual_official_counts
            and actual_official_counts == AOD4_V6_OFFICIAL_COUNTS
        )
    ):
        raise ValueError("Official AOD-4 v6 count gate is incomplete or inconsistent")

    decode_check = metadata["image_decode_check"]
    raw_image_count = sum(len(raw_by_split[split]["images"]) for split in SPLITS)
    if (
        not isinstance(decode_check, dict)
        or decode_check.get("enabled") is not True
        or decode_check.get("method") != "pillow_verify_then_full_pixel_load"
        or int(decode_check.get("images_checked", -1)) != raw_image_count
        or int(decode_check.get("full_pixel_decodes", -1)) != raw_image_count
        or int(decode_check.get("dimension_mismatches", -1)) != 0
    ):
        raise ValueError("Manifest full-pixel image decode/dimension audit is incomplete")

    print("[Data] Preflight is fully rehashing every raw source image...")
    current_inventory = build_source_hash_inventory(
        raw_by_split,
        args.data_root,
        annotation_hashes=annotation_hashes,
        force_rehash=True,
    )
    if current_inventory != stored_inventory:
        raise ValueError(
            "Raw source hash inventory differs from the immutable split manifest"
        )

    actual_source_counts = {}
    actual_raw_source_counts = {}
    for split in SPLITS:
        records = current_inventory["records"][split]
        actual_source_counts[split] = {
            "images": len(canonical_by_split[split]["images"]),
            "annotations": len(canonical_by_split[split]["annotations"]),
        }
        actual_raw_source_counts[split] = {
            "images": len(raw_by_split[split]["images"]),
            "annotations": len(raw_by_split[split]["annotations"]),
            "roboflow_source_keys": len({
                source_image_key(str(record[1])) for record in records
            }),
            "unique_content_sha256": len({str(record[2]).lower() for record in records}),
        }
    if actual_source_counts != stored_source_counts:
        raise ValueError(
            "Canonical pre-purge source counts differ from the manifest: "
            f"current={actual_source_counts}, stored={stored_source_counts}"
        )
    if actual_raw_source_counts != stored_raw_source_counts:
        raise ValueError(
            "Raw source counts/identity cardinalities differ from the manifest: "
            f"current={actual_raw_source_counts}, stored={stored_raw_source_counts}"
        )

    selected_by_split, current_source_audit = apply_source_split_policy(
        canonical_by_split,
        current_inventory,
        policy=source_split_policy,
    )
    if current_source_audit != stored_source_audit:
        raise ValueError(
            "Recomputed cross-split source audit differs from the manifest audit"
        )
    if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL:
        if int(current_source_audit.get("excluded", {}).get("images", -1)) != 0:
            raise ValueError("Official AOD-4 v6 policy excluded source images")
        for split in SPLITS:
            if (
                current_source_audit["before"]["per_split"][split]
                != current_source_audit["after"]["per_split"][split]
            ):
                raise ValueError(
                    f"Official AOD-4 v6 membership changed for {split}"
                )

    inventory_by_split = {
        split: {
            int(record[0]): (str(record[1]), str(record[2]).lower())
            for record in current_inventory["records"][split]
        }
        for split in SPLITS
    }
    seen_group_split = {}
    identity_splits = {
        "filename": {}, "source_key": {}, "sha256": {}, "source_group": {},
    }
    selected_inventory_records = {split: [] for split in SPLITS}
    per_split_source_groups = {}
    for split in SPLITS:
        split_groups = set()
        for image in selected_by_split[split]["images"]:
            image_id = int(image["id"])
            group_id = image.get(SOURCE_GROUP_FIELD)
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(
                    f"{split}: selected image {image.get('id')} has no source group id"
                )
            split_groups.add(group_id)
            previous = seen_group_split.setdefault(group_id, split)
            if (
                source_split_policy == SOURCE_SPLIT_POLICY_EXCLUSIVE
                and previous != split
            ):
                raise ValueError(
                    f"Source group {group_id} crosses exclusive splits {previous}/{split}"
                )
            try:
                file_name, content_digest = inventory_by_split[split][image_id]
            except KeyError as error:
                raise ValueError(
                    f"{split}: selected image {image_id} is absent from raw inventory"
                ) from error
            selected_inventory_records[split].append(
                [image_id, file_name, content_digest]
            )
            identities = {
                "filename": file_name.lower(),
                "source_key": source_image_key(file_name),
                "sha256": content_digest,
                "source_group": group_id,
            }
            for identity_type, identity in identities.items():
                identity_splits[identity_type].setdefault(identity, set()).add(split)
        per_split_source_groups[split] = len(split_groups)

    duplicate_counts = {
        identity_type: sum(len(splits) > 1 for splits in split_sets.values())
        for identity_type, split_sets in identity_splits.items()
    }
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_EXCLUSIVE
        and any(duplicate_counts.values())
    ):
        raise ValueError(
            "Source-exclusive policy retained cross-split identities: "
            f"{duplicate_counts}"
        )

    def _tree_digest(records: list) -> str:
        digest = hashlib.sha256()
        for _, file_name, content_digest in sorted(
            records, key=lambda row: (str(row[1]), int(row[0]))
        ):
            digest.update(str(file_name).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(content_digest).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    recomputed_post_policy = {
        "schema_version": 1,
        "policy": source_split_policy,
        "identity_policy": SOURCE_IDENTITY_POLICY,
        "filename_cross_split_duplicates": duplicate_counts["filename"],
        "roboflow_source_key_cross_split_duplicates": duplicate_counts["source_key"],
        "sha256_cross_split_duplicates": duplicate_counts["sha256"],
        "source_group_cross_split_duplicates": duplicate_counts["source_group"],
        "per_split_source_groups": per_split_source_groups,
        "per_split_image_tree_sha256": dict(
            current_inventory["per_split_image_tree_sha256"]
        ),
        "selected_per_split_image_tree_sha256": {
            split: _tree_digest(selected_inventory_records[split])
            for split in SPLITS
        },
        "source_hash_inventory_sha256": current_inventory["inventory_sha256"],
    }
    if recomputed_post_policy != metadata["post_policy_cross_split_source_check"]:
        raise ValueError(
            "Recomputed post-policy source check differs from the manifest"
        )

    coco_by_split = {
        split: filter_annotations(
            selected_by_split[split],
            args.min_bbox_area,
            args.min_bbox_side,
            drop_empty_images=False,
        )
        for split in SPLITS
    }
    for split in SPLITS:
        actual_counts = {
            "images": len(coco_by_split[split]["images"]),
            "annotations": len(coco_by_split[split]["annotations"]),
        }
        stored_counts = metadata["split_counts"][split]
        normalized_stored = {
            key: int(stored_counts[key]) for key in actual_counts
        }
        if actual_counts != normalized_stored:
            raise ValueError(
                f"{split}: selected split_counts={actual_counts} differ from "
                f"manifest={normalized_stored}"
            )
        if (
            source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
            and actual_counts != actual_source_counts[split]
        ):
            raise ValueError(
                f"{split}: official AOD-4 v6 count preservation failed"
            )

    cat_ids = sorted(expected_target_categories)
    class_names = [expected_target_categories[cat_id] for cat_id in cat_ids]
    expected_cat_to_label = {
        str(cat_id): index for index, cat_id in enumerate(cat_ids)
    }
    stored_cat_to_label = {
        str(key): int(value) for key, value in metadata["cat_id_to_label"].items()
    }
    if class_names != [str(value) for value in metadata["class_names"]]:
        raise ValueError("Manifest class_names differ from canonical source categories")
    if expected_cat_to_label != stored_cat_to_label:
        raise ValueError("Manifest category-to-label mapping differs from canonical source")
    if len(class_names) != int(args.num_classes):
        raise ValueError(
            f"COCO has {len(class_names)} classes but --num_classes={args.num_classes}"
        )

    expected_proportions = draw_client_proportions(
        cat_ids,
        args.num_clients,
        args.partition,
        args.dirichlet_alpha,
        args.partition_seed,
    )
    stored_proportions = metadata["client_target_proportions"]
    if set(stored_proportions) != set(expected_proportions):
        raise ValueError("Manifest client_target_proportions has wrong class keys")
    for key, expected_values in expected_proportions.items():
        try:
            actual_values = np.asarray(stored_proportions[key], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Manifest target proportions for {key} are not numeric"
            ) from error
        if actual_values.shape != (args.num_clients,) or not np.allclose(
            actual_values,
            np.asarray(expected_values),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                f"Manifest target proportions for {key} differ from seeded draw"
            )

    clients = _client_records(manifest, args.num_clients)
    assignments = {}
    split_seed_offsets = {"train": 0, "val": 1001, "test": 2001}
    initial_canonical_assignments = {}
    for split in SPLITS:
        split_seed = args.partition_seed + split_seed_offsets[split]
        if args.partition == "iid":
            initial_canonical_assignments[split] = partition_images_iid(
                coco_by_split[split],
                args.num_clients,
                split_seed,
                group_atomic=True,
            )
        else:
            initial_canonical_assignments[split] = partition_images(
                coco_by_split[split],
                args.num_clients,
                expected_proportions,
                split_seed,
                group_atomic=True,
            )
    canonical_assignments_by_split, current_cross_client_report = (
        align_cross_split_source_group_owners(
            coco_by_split,
            initial_canonical_assignments,
        )
    )
    if current_cross_client_report != metadata.get(
        "cross_split_client_source_group_check"
    ):
        raise ValueError(
            "Cross-split source-group client ownership differs from manifest"
        )
    for split in SPLITS:
        per_client = []
        for client in clients:
            block = client.get("splits", {}).get(split)
            if not isinstance(block, dict):
                raise ValueError(
                    f"Client {client['client_id']} has no {split} split block"
                )
            image_ids = [int(value) for value in block.get("image_ids", [])]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(
                    f"Client {client['client_id']} {split} has duplicate image IDs"
                )
            if len(image_ids) != int(block.get("num_images", -1)):
                raise ValueError(
                    f"Client {client['client_id']} {split} num_images mismatch"
                )
            per_client.append(image_ids)

        expected_ids = {
            int(image["id"]) for image in coco_by_split[split]["images"]
        }
        flat = [image_id for image_ids in per_client for image_id in image_ids]
        if len(flat) != len(set(flat)) or set(flat) != expected_ids:
            raise ValueError(
                f"{split}: client assignments are not an exact cleaned-image cover"
            )
        group_summary = validate_source_group_assignments(
            coco_by_split[split], per_client
        )
        for client, expected_group_count in zip(
            clients, group_summary["client_source_group_counts"]
        ):
            stored_group_count = int(
                client["splits"][split].get("num_source_groups", -1)
            )
            if stored_group_count != int(expected_group_count):
                raise ValueError(
                    f"Client {client['client_id']} {split} source-group count mismatch"
                )

        canonical_assignments = canonical_assignments_by_split[split]
        if per_client != canonical_assignments:
            raise ValueError(
                f"{split}: manifest assignment differs from deterministic "
                "source-group partition"
            )

        sizes = [len(values) for values in per_client]
        expected_balance = {
            "min_images": min(sizes),
            "max_images": max(sizes),
            "max_min_gap": max(sizes) - min(sizes),
            "min_source_groups": min(
                group_summary["client_source_group_counts"]
            ),
            "max_source_groups": max(
                group_summary["client_source_group_counts"]
            ),
            "largest_source_group_images": group_summary[
                "largest_source_group_images"
            ],
            "balance_bound_images": group_summary[
                "largest_source_group_images"
            ],
        }
        stored_balance = metadata["quantity_balance"][split]
        if any(
            int(stored_balance.get(key, -1)) != value
            for key, value in expected_balance.items()
        ):
            raise ValueError(
                f"{split}: recorded quantity-balance statistics are inconsistent"
            )
        if stored_balance.get("source_group_atomic") is not True:
            raise ValueError(
                f"{split}: manifest does not mark source-group atomic assignment"
            )
        if stored_balance.get("balance_bound_satisfied") is not True:
            raise ValueError(
                f"{split}: manifest does not satisfy its source-group balance bound"
            )
        if group_summary["max_min_image_gap"] > group_summary[
            "largest_source_group_images"
        ]:
            raise ValueError(
                f"{split}: client image-count gap exceeds the largest source group"
            )

        recomputed = split_statistics(coco_by_split[split], per_client)
        stored_realized = metadata["realized_partition_statistics"].get(split)
        if stored_realized != recomputed:
            raise ValueError(
                f"{split}: top-level realized partition statistics differ from "
                "the cleaned client assignments"
            )
        for client, row in zip(clients, recomputed):
            stored = client["splits"][split]
            stored_instances = {
                str(key): int(value)
                for key, value in stored["class_instances"].items()
            }
            stored_class_images = {
                str(key): int(value)
                for key, value in stored["class_images"].items()
            }
            if stored_instances != row["class_instances"]:
                raise ValueError(
                    f"Client {client['client_id']} {split} class histogram mismatch"
                )
            if stored_class_images != row["class_images"]:
                raise ValueError(
                    f"Client {client['client_id']} {split} class-image histogram mismatch"
                )
            if int(stored["background_images"]) != row["background_images"]:
                raise ValueError(
                    f"Client {client['client_id']} {split} background count mismatch"
                )
            if int(stored["num_images"]) != row["num_images"]:
                raise ValueError(
                    f"Client {client['client_id']} {split} image count mismatch"
                )
            if int(stored["minimum_class_instances"]) != row[
                "minimum_class_instances"
            ]:
                raise ValueError(
                    f"Client {client['client_id']} {split} minimum-class count mismatch"
                )
            for key in ("class_entropy_nats", "js_divergence_from_global_nats"):
                if not math.isclose(
                    float(stored[key]), float(row[key]),
                    rel_tol=1e-12, abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Client {client['client_id']} {split} {key} mismatch"
                    )
        assignments[split] = per_client

    _pass(
        "schema-v7 raw inventory, selected AOD-4 source policy, exact covers, "
        "and split-local source-group-atomic client assignments"
    )
    return manifest, coco_by_split, assignments


def _relative_file_set(root: str, suffix: str | None = None) -> set:
    values = set()
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if suffix is not None and not filename.lower().endswith(suffix.lower()):
                continue
            path = os.path.join(directory, filename)
            values.add(os.path.relpath(path, root).replace(os.sep, "/"))
    return values


def _expected_label_rows(coco: dict, image_id: int, cat_to_label: Dict[int, int]):
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    anns_by_image = build_image_ann_map(coco)
    image = image_by_id[image_id]
    rows = []
    for annotation in anns_by_image.get(image_id, []):
        if annotation.get("iscrowd", 0):
            continue
        category_id = int(annotation["category_id"])
        if category_id not in cat_to_label:
            raise ValueError(f"Unmapped category {category_id} while validating YOLO labels")
        clipped = _clip_bbox_xywh(
            annotation["bbox"],
            int(image["width"]),
            int(image["height"]),
        )
        if clipped is not None:
            rows.append((cat_to_label[category_id], *clipped))
    return rows


def _parse_label(path: str, num_classes: int) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields, got {len(fields)}")
            try:
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid YOLO number") from error
            values = (cx, cy, width, height)
            if not 0 <= class_id < num_classes:
                raise ValueError(f"{path}:{line_number}: class {class_id} outside 0..{num_classes - 1}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite coordinate")
            if width <= 0 or height <= 0:
                raise ValueError(f"{path}:{line_number}: width/height must be positive")
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"{path}:{line_number}: normalized coordinate outside [0,1]")
            tolerance = 1e-7
            if (
                cx - width / 2 < -tolerance or cx + width / 2 > 1 + tolerance
                or cy - height / 2 < -tolerance or cy + height / 2 > 1 + tolerance
            ):
                raise ValueError(f"{path}:{line_number}: decoded box exceeds image bounds")
            rows.append((class_id, cx, cy, width, height))
    return rows


def _validate_dataset_yaml(
    yaml_path: str,
    expected_ids: Dict[str, Iterable[int]],
    coco_by_split: Dict[str, dict],
    data_root: str,
    yolo_root: str,
    class_names: List[str],
    cat_to_label: Dict[int, int],
):
    payload = _load_yaml(yaml_path)
    required = {"path", "train", "val", "test", "nc", "names"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{yaml_path}: missing YAML keys {sorted(missing)}")
    names = _normalise_names(payload["names"])
    if int(payload["nc"]) != len(class_names) or names != class_names:
        raise ValueError(
            f"{yaml_path}: nc/names {payload['nc']}/{names} != expected {len(class_names)}/{class_names}"
        )
    yaml_base = os.path.dirname(os.path.abspath(yaml_path))
    yaml_root = os.path.abspath(payload["path"])
    if yaml_root != yaml_base:
        raise ValueError(f"{yaml_path}: path={yaml_root} must equal YAML directory {yaml_base}")
    if os.path.commonpath([yaml_root, os.path.abspath(yolo_root)]) != os.path.abspath(yolo_root):
        raise ValueError(f"{yaml_path}: dataset path escapes generated YOLO root")

    for split in SPLITS:
        image_dir = str(payload[split])
        if not os.path.isabs(image_dir):
            image_dir = os.path.join(yaml_root, image_dir)
        image_dir = os.path.abspath(image_dir)
        label_dir = os.path.join(os.path.dirname(image_dir), "labels")
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"{yaml_path}: missing {split} images/labels directory: {image_dir}, {label_dir}"
            )

        image_by_id = {int(image["id"]): image for image in coco_by_split[split]["images"]}
        selected_ids = [int(value) for value in expected_ids[split]]
        selected_names = [str(image_by_id[image_id]["file_name"]).replace("\\", "/")
                          for image_id in selected_ids]
        if len(selected_names) != len(set(selected_names)):
            raise ValueError(f"{yaml_path}: selected {split} file names are not unique")
        expected_images = set(selected_names)
        expected_labels = {os.path.splitext(name)[0] + ".txt" for name in selected_names}
        actual_images = _relative_file_set(image_dir)
        actual_labels = _relative_file_set(label_dir, suffix=".txt")
        if actual_images != expected_images:
            raise ValueError(
                f"{yaml_path} {split}: image tree mismatch; "
                f"missing={sorted(expected_images - actual_images)[:10]}, "
                f"stale={sorted(actual_images - expected_images)[:10]}"
            )
        if actual_labels != expected_labels:
            raise ValueError(
                f"{yaml_path} {split}: label tree mismatch; "
                f"missing={sorted(expected_labels - actual_labels)[:10]}, "
                f"stale={sorted(actual_labels - expected_labels)[:10]}"
            )

        for image_id, relative_name in zip(selected_ids, selected_names):
            generated_image = os.path.join(image_dir, *relative_name.split("/"))
            expected_source = os.path.join(data_root, split, *relative_name.split("/"))
            if not os.path.islink(generated_image):
                raise ValueError(f"Generated image is not a symlink: {generated_image}")
            if os.path.realpath(generated_image) != os.path.realpath(expected_source):
                raise ValueError(
                    f"Generated symlink target mismatch: {generated_image} -> "
                    f"{os.path.realpath(generated_image)}, expected {expected_source}"
                )
            label_name = os.path.splitext(relative_name)[0] + ".txt"
            label_path = os.path.join(label_dir, *label_name.split("/"))
            actual_rows = _parse_label(label_path, len(class_names))
            expected_rows = _expected_label_rows(coco_by_split[split], image_id, cat_to_label)
            if len(actual_rows) != len(expected_rows):
                raise ValueError(
                    f"{label_path}: {len(actual_rows)} rows != expected {len(expected_rows)}"
                )
            for row_index, (actual, expected) in enumerate(zip(actual_rows, expected_rows), start=1):
                if actual[0] != expected[0] or not np.allclose(
                    actual[1:], expected[1:], rtol=0.0, atol=5e-8
                ):
                    raise ValueError(
                        f"{label_path}:{row_index}: label {actual} != clipped COCO {expected}"
                    )


def validate_yolo_tree(args, manifest: dict, coco_by_split: Dict[str, dict],
                       assignments: Dict[str, List[List[int]]]):
    metadata = manifest["metadata"]
    class_names = [str(value) for value in metadata["class_names"]]
    cat_to_label = {int(key): int(value) for key, value in metadata["cat_id_to_label"].items()}
    yolo_root = _derive_yolo_dir(args.split_file)
    if not os.path.isdir(yolo_root):
        raise FileNotFoundError(f"Generated YOLO directory not found: {yolo_root}")
    current_tree_digest = generated_yolo_tree_sha256(yolo_root)
    if current_tree_digest != metadata["generated_yolo_tree_sha256"]:
        raise ValueError("Generated YOLO tree digest differs from the split manifest")

    for client_id in range(args.num_clients):
        yaml_path = os.path.join(yolo_root, f"client_{client_id}", "dataset.yaml")
        expected = {split: assignments[split][client_id] for split in SPLITS}
        _validate_dataset_yaml(
            yaml_path, expected, coco_by_split, args.data_root, yolo_root,
            class_names, cat_to_label,
        )

    full_yaml = os.path.join(yolo_root, "full", "dataset.yaml")
    expected_full = {
        split: [int(image["id"]) for image in coco_by_split[split]["images"]]
        for split in SPLITS
    }
    _validate_dataset_yaml(
        full_yaml, expected_full, coco_by_split, args.data_root, yolo_root,
        class_names, cat_to_label,
    )
    _pass("every client/full YAML, symlink, background label and clipped YOLO row matches COCO")


def validate_model_structure(wrapper: RTDETRLoRA, args):
    """Check exact adapter roles, freeze policy and strict FL state round-trip."""
    adapters = [
        (name, module) for name, module in wrapper.model.named_modules()
        if isinstance(module, (LoRALinear, LoRAConv2d, LoRAMultiheadAttention))
    ]
    if args.fl_method == "full_ft":
        if adapters:
            raise RuntimeError("Full fine-tuning model unexpectedly contains LoRA adapters")
        frozen = [name for name, parameter in wrapper.model.named_parameters()
                  if not parameter.requires_grad]
        if frozen:
            raise RuntimeError(f"Full fine-tuning has frozen parameters: {frozen[:10]}")
    else:
        if not adapters:
            raise RuntimeError("LoRA method has no injected adapters")
        names = [name for name, _ in adapters]
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate adapter names in model manifest")

        mha = [(name, module) for name, module in adapters
               if isinstance(module, LoRAMultiheadAttention)]
        linear = [(name, module) for name, module in adapters if isinstance(module, LoRALinear)]
        conv = [(name, module) for name, module in adapters if isinstance(module, LoRAConv2d)]
        if args.apply_lora_decoder:
            if not mha or not linear:
                raise RuntimeError("Decoder LoRA requires fused Q/K/V MHA and deformable value adapters")
        elif mha or linear:
            raise RuntimeError("Decoder adapters exist despite --no-apply_lora_decoder")
        if args.apply_lora_backbone and not conv:
            raise RuntimeError("Backbone LoRA was requested but no Conv2d adapter exists")
        if not args.apply_lora_backbone and conv:
            raise RuntimeError("Backbone adapters exist despite --no-apply_lora_backbone")

        for name, _ in mha:
            if not (".decoder.layers." in name and name.endswith(".self_attn")):
                raise RuntimeError(f"MHA adapter is outside decoder self-attention: {name}")
        for name, _ in linear:
            if not name.endswith(".cross_attn.value_proj"):
                raise RuntimeError(f"Linear adapter is outside deformable value projection: {name}")
        backbone_len = len(wrapper.model.yaml["backbone"])
        backbone_prefixes = tuple(f"model.{index}" for index in range(backbone_len))
        for name, _ in conv:
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in backbone_prefixes):
                raise RuntimeError(f"Conv adapter is outside declared CNN backbone: {name}")

        unexpected_trainable = []
        for name, parameter in wrapper.model.named_parameters():
            allowed = (
                name.endswith(".lora_A")
                or name.endswith(".lora_B")
                or wrapper._is_task_head_key(name)
            )
            if parameter.requires_grad != allowed:
                unexpected_trainable.append((name, parameter.requires_grad, allowed))
        if unexpected_trainable:
            raise RuntimeError(
                "LoRA freeze policy mismatch (name, requires_grad, expected): "
                f"{unexpected_trainable[:10]}"
            )
        for name, module in adapters:
            if torch.count_nonzero(module.lora_B.detach()).item() != 0:
                raise RuntimeError(f"Adapter B must be zero-initialized: {name}")

    state = wrapper.get_aggregation_state()
    if not state or any(value.device.type != "cpu" for value in state.values()):
        raise RuntimeError("Aggregation state must be a non-empty CPU tensor mapping")
    if adapters:
        shared_factor_roles = set()
        for key in state:
            if key.endswith(".lora_A"):
                shared_factor_roles.add("A")
            elif key.endswith(".lora_B"):
                shared_factor_roles.add("B")
        expected_shared_roles = {
            "lora": {"A", "B"},
            "fedsa_lora": {"A"},
            "fixed_share_b_lora": {"B"},
        }[args.fl_method]
        if shared_factor_roles != expected_shared_roles:
            raise RuntimeError(
                "Shared LoRA factor routing mismatch: "
                f"observed={sorted(shared_factor_roles)}, "
                f"expected={sorted(expected_shared_roles)}"
            )
        if not any(wrapper._is_task_head_key(key) for key in state):
            raise RuntimeError("LoRA federated payload has no globally shared task head")

    before_local = wrapper.get_local_personalized_state()
    local_role = wrapper.local_personalized_factor_role()
    expected_local_suffix = f".lora_{local_role}" if local_role is not None else None
    if expected_local_suffix is None:
        if before_local:
            raise RuntimeError("Non-personalized method exposed a client-local LoRA state")
    elif not before_local or any(
        not key.endswith(expected_local_suffix) for key in before_local
    ):
        raise RuntimeError(
            f"Client-local state does not contain only LoRA factor {local_role}"
        )
    wrapper.set_aggregation_state(state)
    after_local = wrapper.get_local_personalized_state()
    if set(before_local) != set(after_local) or any(
        not torch.equal(before_local[key], after_local[key]) for key in before_local
    ):
        raise RuntimeError(
            f"Global state load modified client-local LoRA factor {local_role}"
        )
    counts = wrapper.count_params()
    payload_numel = sum(value.numel() for value in state.values())
    payload_bytes = sum(value.numel() * value.element_size() for value in state.values())
    if payload_numel != counts["communication_params"]:
        raise RuntimeError("Reported communication parameter count differs from actual payload")
    if payload_bytes != counts["communication_bytes"]:
        raise RuntimeError("Reported communication byte count differs from actual payload")
    _pass(
        f"model head/freeze/state policy and exact adapter targets "
        f"(MHA={sum(isinstance(m, LoRAMultiheadAttention) for _, m in adapters)}, "
        f"linear={sum(isinstance(m, LoRALinear) for _, m in adapters)}, "
        f"conv={sum(isinstance(m, LoRAConv2d) for _, m in adapters)})"
    )


def _probe_dataset_yaml(args, yolo_root: str) -> str:
    if args.mode == "centralized":
        return os.path.join(yolo_root, "full", "dataset.yaml")
    client_id = args.client_id if args.mode == "solo" else 0
    if client_id is None or not 0 <= int(client_id) < int(args.num_clients):
        raise ValueError(f"Invalid probe client_id={client_id}")
    return os.path.join(yolo_root, f"client_{int(client_id)}", "dataset.yaml")


def _foreground_index(dataset) -> int:
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, list):
        for index, label in enumerate(labels):
            classes = np.asarray(label.get("cls", [])) if isinstance(label, dict) else np.asarray([])
            if classes.size:
                return index
    if len(dataset) == 0:
        raise RuntimeError("RT-DETR dataset is empty")
    print("[WARN] No foreground sample found; probing a background-only loss batch")
    return 0


def validate_one_loss_batch(wrapper: RTDETRLoRA, args, yolo_root: str):
    """Run one deterministic labeled loss/backward batch and audit gradients."""
    yaml_path = _probe_dataset_yaml(args, yolo_root)
    dataset = build_rtdetr_dataset(yaml_path, "train", args, augment=False)
    index = _foreground_index(dataset)
    loader = DataLoader(
        Subset(dataset, [index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    batch = next(iter(loader))
    required = {"img", "cls", "bboxes", "batch_idx"}
    missing = required - set(batch)
    if missing:
        raise RuntimeError(f"RT-DETR batch is missing keys: {sorted(missing)}")
    if batch["img"].ndim != 4 or batch["img"].shape[0] != 1:
        raise RuntimeError(f"Unexpected image batch shape: {tuple(batch['img'].shape)}")
    if not torch.isfinite(batch["bboxes"]).all():
        raise RuntimeError("RT-DETR batch contains non-finite boxes")
    if batch["bboxes"].numel() and not (
        (batch["bboxes"] >= 0).all() and (batch["bboxes"] <= 1).all()
    ):
        raise RuntimeError("RT-DETR batch boxes are not normalized to [0,1]")

    device = torch.device(args.device)
    wrapper.model.to(device)
    wrapper.model.train()
    wrapper.enforce_frozen_norm_eval()
    wrapper.model.zero_grad(set_to_none=True)
    moved = _move_batch(batch, device)
    output = wrapper.model(moved)
    loss, components = _unpack_loss(output)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite RT-DETR preflight loss: {float(loss.detach())}")
    loss.backward()

    trainable = [(name, parameter) for name, parameter in wrapper.model.named_parameters()
                 if parameter.requires_grad]
    finite_gradients = []
    disconnected = []
    nonzero = []
    for name, parameter in trainable:
        if parameter.grad is None:
            disconnected.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient for {name}")
        finite_gradients.append(name)
        if torch.count_nonzero(parameter.grad.detach()).item() > 0:
            nonzero.append(name)
    if not finite_gradients or not nonzero:
        raise RuntimeError("One-batch backward produced no finite nonzero trainable gradients")

    adapters = [(name, module) for name, module in wrapper.model.named_modules()
                if isinstance(module, (LoRALinear, LoRAConv2d, LoRAMultiheadAttention))]
    if adapters:
        adapter_disconnected = []
        nonzero_b_types = set()
        for name, module in adapters:
            if module.lora_A.grad is None or module.lora_B.grad is None:
                adapter_disconnected.append(name)
                continue
            if not torch.isfinite(module.lora_A.grad).all() or not torch.isfinite(module.lora_B.grad).all():
                raise FloatingPointError(f"Non-finite adapter gradient: {name}")
            # B=0 makes the first A gradient exactly zero; this is the intended
            # FedSA/LoRA initialization, not a disconnected computation graph.
            if torch.count_nonzero(module.lora_A.grad.detach()).item() != 0:
                raise RuntimeError(f"Initial A gradient should be zero while B=0: {name}")
            if torch.count_nonzero(module.lora_B.grad.detach()).item() > 0:
                nonzero_b_types.add(type(module).__name__)
        if adapter_disconnected:
            raise RuntimeError(f"Injected adapters are disconnected from loss: {adapter_disconnected[:10]}")
        expected_types = {type(module).__name__ for _, module in adapters}
        if nonzero_b_types != expected_types:
            raise RuntimeError(
                f"No nonzero first-step B gradient for adapter type(s): "
                f"{sorted(expected_types - nonzero_b_types)}"
            )

    task_head_nonzero = [name for name in nonzero if wrapper._is_task_head_key(name)]
    if not task_head_nonzero:
        raise RuntimeError("Globally shared task head has no nonzero gradient")
    # Non-adapter trainables may be branch-dependent, but adapter disconnection
    # and all non-finite gradients are always fatal. Report any other omissions.
    non_adapter_disconnected = [
        name for name in disconnected
        if ".lora_A" not in name and ".lora_B" not in name
    ]
    if non_adapter_disconnected:
        print(f"[WARN] Branch-dependent trainables without this batch's gradient: {non_adapter_disconnected[:10]}")

    component_values = {
        key: float(value.detach().cpu()) for key, value in components.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }
    _pass(
        f"one RT-DETR labeled forward/backward batch on {device}; "
        f"loss={float(loss.detach().cpu()):.6f}, components={component_values}"
    )


def main(argv=None):
    args = get_args(argv)
    print("=" * 72)
    print("AOD-4 RT-DETR experiment preflight (no training/checkpoint writes)")
    print(f"data_root:  {os.path.abspath(args.data_root)}")
    print(f"split_file: {os.path.abspath(args.split_file)}")
    print(f"mode/method: {args.mode}/{args.fl_method}")
    print(f"device: {args.device} (AMP intentionally disabled for the smoke loss)")
    print("=" * 72)

    manifest, coco_by_split, assignments = validate_manifest(args)
    yolo_root = _derive_yolo_dir(args.split_file)
    validate_yolo_tree(args, manifest, coco_by_split, assignments)

    wrapper = RTDETRLoRA(args, class_names=list(manifest["metadata"]["class_names"]))
    validate_model_structure(wrapper, args)
    validate_one_loss_batch(wrapper, args, yolo_root)
    _pass("all preflight checks completed; the experiment is ready to launch")


if __name__ == "__main__":
    main()
