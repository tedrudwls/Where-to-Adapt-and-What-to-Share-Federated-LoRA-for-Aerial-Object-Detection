#!/usr/bin/env python3
"""Generate immutable AOD-4 client train/val/test partitions and YOLO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_DATA_ROOT
from data.dataset import (
    AOD4_CATEGORY_POLICY,
    AOD4_TARGET_CLASSES,
    AOD4_V6_OFFICIAL_COUNTS,
    CLIENT_PARTITION_UNIT,
    DEFAULT_SOURCE_SPLIT_POLICY,
    SOURCE_GROUP_FIELD,
    SOURCE_IDENTITY_POLICY,
    SOURCE_SPLIT_POLICY_EXCLUSIVE,
    SOURCE_SPLIT_POLICY_OFFICIAL,
    SOURCE_SPLIT_PRIORITY,
    SUPPORTED_SOURCE_SPLIT_POLICIES,
    SPLITS,
    annotation_sha256,
    align_cross_split_source_group_owners,
    apply_source_split_policy,
    build_source_hash_inventory,
    canonicalize_aod4_categories,
    category_mapping,
    coco_to_yolo_labels,
    create_dataset_yaml,
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


# Compatibility for callers that imported the historical script-local helper.
# The canonical implementation now lives in ``data.dataset`` so preparation,
# runtime validation and the read-only checker cannot drift.
_source_image_key = source_image_key


def _split_tag(args) -> str:
    policy_tag = (
        "official_v6"
        if args.source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        else "source_exclusive"
    )
    if args.partition == "iid":
        tag = f"{policy_tag}_iid_c{args.num_clients}_s{args.seed}"
    else:
        tag = (
            f"{policy_tag}_dirichlet_a{args.dirichlet_alpha:g}_"
            f"c{args.num_clients}_s{args.seed}"
        )
    if args.min_bbox_area > 0 or args.min_bbox_side > 0:
        tag += f"_area{args.min_bbox_area:g}_side{args.min_bbox_side:g}"
    return tag


def _verify_image_decoding(coco_by_split: dict, data_root: str, enabled: bool) -> dict:
    """Decode every source image header/body and verify COCO pixel dimensions."""
    report = {
        "enabled": bool(enabled),
        "method": "pillow_verify_then_full_pixel_load",
        "images_checked": 0,
        "full_pixel_decodes": 0,
        "dimension_mismatches": 0,
    }
    if not enabled:
        return report
    failures = []
    mismatches = []
    for split, coco in coco_by_split.items():
        for image in coco["images"]:
            name = str(image["file_name"])
            path = os.path.join(data_root, split, name)
            try:
                with Image.open(path) as decoded:
                    actual_size = tuple(map(int, decoded.size))
                    decoded.verify()
                # Pillow.verify() checks container integrity but can accept a
                # JPEG whose entropy-coded pixel body is truncated. Reopen and
                # force a complete decode so DataLoader workers cannot discover
                # that corruption only after a long run has started.
                with Image.open(path) as decoded:
                    decoded.load()
                report["full_pixel_decodes"] += 1
            except Exception as error:
                failures.append((split, name, type(error).__name__, str(error)))
                continue
            expected_size = (int(image["width"]), int(image["height"]))
            if actual_size != expected_size:
                mismatches.append((split, name, expected_size, actual_size))
            report["images_checked"] += 1
    report["dimension_mismatches"] = len(mismatches)
    if failures:
        preview = "\n".join(map(str, failures[:10]))
        raise ValueError(
            f"{len(failures)} source images cannot be decoded. First entries:\n{preview}"
        )
    if mismatches:
        preview = "\n".join(map(str, mismatches[:10]))
        raise ValueError(
            f"{len(mismatches)} COCO image dimensions differ from decoded files. "
            f"First entries:\n{preview}"
        )
    return report


def _inventory_maps(hash_inventory: dict) -> dict:
    """Return ``split -> image_id -> (filename, sha256)`` from the inventory."""
    records = hash_inventory.get("records")
    if not isinstance(records, dict) or set(records) != set(SPLITS):
        raise ValueError("Source hash inventory must cover train/val/test")
    result = {}
    for split in SPLITS:
        split_map = {}
        for row in records[split]:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{split}: malformed source hash record {row!r}")
            image_id, filename, digest = int(row[0]), str(row[1]), str(row[2])
            if image_id in split_map:
                raise ValueError(f"{split}: duplicate image id {image_id} in hash inventory")
            split_map[image_id] = (filename, digest)
        result[split] = split_map
    return result


def _tree_digest_from_inventory_records(records: list) -> str:
    digest = hashlib.sha256()
    for image_id, filename, content_digest in sorted(
        records, key=lambda row: (str(row[1]), int(row[0]))
    ):
        del image_id
        digest.update(str(filename).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_digest).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _raw_source_counts(coco_by_split: dict, hash_inventory: dict) -> dict:
    inventory_maps = _inventory_maps(hash_inventory)
    result = {}
    for split in SPLITS:
        images = coco_by_split[split]["images"]
        source_keys = set()
        content_hashes = set()
        for image in images:
            image_id = int(image["id"])
            filename, content_digest = inventory_maps[split][image_id]
            source_keys.add(source_image_key(filename))
            content_hashes.add(content_digest)
        result[split] = {
            "images": len(images),
            "annotations": len(coco_by_split[split]["annotations"]),
            "roboflow_source_keys": len(source_keys),
            "unique_content_sha256": len(content_hashes),
        }
    return result


def _check_post_policy_sources(selected_by_split: dict,
                               hash_inventory: dict,
                               source_split_policy: str) -> dict:
    """Verify selected source records and report, rather than hide, overlap."""
    inventory_maps = _inventory_maps(hash_inventory)
    seen = {
        "filename": {},
        "source_key": {},
        "sha256": {},
        "source_group": {},
    }
    identity_splits = {key: {} for key in seen}
    selected_records = {split: [] for split in SPLITS}
    per_split_source_groups = {}
    for split in SPLITS:
        groups = set()
        for image in selected_by_split[split]["images"]:
            image_id = int(image["id"])
            if image_id not in inventory_maps[split]:
                raise ValueError(
                    f"{split}: selected image {image_id} is absent from source inventory"
                )
            filename, content_digest = inventory_maps[split][image_id]
            group_id = image.get(SOURCE_GROUP_FIELD)
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(
                    f"{split}: selected image {image_id} has no {SOURCE_GROUP_FIELD}"
                )
            groups.add(group_id)
            selected_records[split].append([image_id, filename, content_digest])
            identities = {
                "filename": filename.lower(),
                "source_key": source_image_key(filename),
                "sha256": content_digest,
                "source_group": group_id,
            }
            for identity_type, identity in identities.items():
                seen[identity_type].setdefault(
                    identity, (split, image_id, filename)
                )
                identity_splits[identity_type].setdefault(identity, set()).add(split)
        per_split_source_groups[split] = len(groups)

    duplicate_counts = {
        identity_type: sum(len(splits) > 1 for splits in split_sets.values())
        for identity_type, split_sets in identity_splits.items()
    }
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_EXCLUSIVE
        and any(duplicate_counts.values())
    ):
        raise RuntimeError(
            "Source-exclusive policy left cross-split collisions: "
            + json.dumps(duplicate_counts, ensure_ascii=False, sort_keys=True)
        )
    return {
        "schema_version": 1,
        "policy": source_split_policy,
        "identity_policy": SOURCE_IDENTITY_POLICY,
        "filename_cross_split_duplicates": duplicate_counts["filename"],
        "roboflow_source_key_cross_split_duplicates": duplicate_counts["source_key"],
        "sha256_cross_split_duplicates": duplicate_counts["sha256"],
        "source_group_cross_split_duplicates": duplicate_counts["source_group"],
        "per_split_source_groups": per_split_source_groups,
        # Runtime immutability checks deliberately hash the complete raw source
        # export, including records excluded by the cleaning policy.
        "per_split_image_tree_sha256": dict(
            hash_inventory["per_split_image_tree_sha256"]
        ),
        "selected_per_split_image_tree_sha256": {
            split: _tree_digest_from_inventory_records(selected_records[split])
            for split in SPLITS
        },
        "source_hash_inventory_sha256": hash_inventory["inventory_sha256"],
    }


def _client_source_group_counts(coco: dict, assignments: list) -> list:
    image_groups = {
        int(image["id"]): str(image[SOURCE_GROUP_FIELD])
        for image in coco["images"]
    }
    owner_by_group = {}
    counts = []
    for client_id, image_ids in enumerate(assignments):
        groups = {image_groups[int(image_id)] for image_id in image_ids}
        for group_id in groups:
            previous = owner_by_group.setdefault(group_id, client_id)
            if previous != client_id:
                raise RuntimeError(
                    f"Source group {group_id} crosses clients {previous} and {client_id}"
                )
        counts.append(len(groups))
    if set(image_groups) != {
        int(image_id) for client in assignments for image_id in client
    }:
        raise RuntimeError("Client assignments are not an exact cleaned-image cover")
    return counts


def prepare_aod4_data(
    data_root: str,
    *,
    min_bbox_area: float = 0.0,
    min_bbox_side: float = 0.0,
    num_classes: int = 4,
    verify_image_hashes: bool = True,
    verify_image_decode: bool = True,
    hash_cache_path: str | None = None,
    source_split_policy: str = DEFAULT_SOURCE_SPLIT_POLICY,
    enforce_official_counts: bool = True,
) -> dict:
    """Run canonical validation, overlap audit, split policy and filtering."""
    if source_split_policy not in SUPPORTED_SOURCE_SPLIT_POLICIES:
        raise ValueError(
            f"Unsupported source split policy {source_split_policy!r}"
        )
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and (min_bbox_area != 0.0 or min_bbox_side != 0.0)
    ):
        raise ValueError(
            "official_aod4_v6 requires zero bbox thresholds so annotations are "
            "preserved exactly"
        )
    if not verify_image_hashes:
        raise ValueError(
            "Publication source auditing requires SHA-256 for every image; "
            "--no-verify_image_hashes is not supported"
        )
    if not verify_image_decode:
        raise ValueError(
            "Publication preprocessing requires decoding every image; "
            "--no-verify_image_decode is not supported"
        )

    data_root = os.path.abspath(data_root)
    print(f"[1/5] Loading and validating raw AOD-4 from {data_root}")
    raw_coco = {}
    canonical_by_split = {}
    validation = {}
    annotation_hashes = {}
    declared_categories = None
    target_categories = None
    source_category_audit = {}
    for split in SPLITS:
        json_path = os.path.join(data_root, split, "_annotations.coco.json")
        raw = load_coco_annotations(json_path)
        current_categories = category_mapping(raw)
        if declared_categories is None:
            declared_categories = current_categories
        validation[split] = validate_coco(
            raw,
            os.path.join(data_root, split),
            split,
            expected_categories=declared_categories,
        )
        canonical, category_audit = canonicalize_aod4_categories(raw)
        current_targets = category_mapping(canonical)
        if target_categories is None:
            target_categories = current_targets
        elif current_targets != target_categories:
            raise ValueError(
                f"{split}: canonical AOD-4 target mapping differs from train: "
                f"{current_targets} != {target_categories}"
            )
        raw_coco[split] = raw
        canonical_by_split[split] = canonical
        source_category_audit[split] = category_audit
        annotation_hashes[split] = annotation_sha256(json_path)
        print(f"  {split}: {validation[split]}")
        ignored = category_audit["ignored_unreferenced_categories"]
        if ignored:
            print(
                "    Ignoring declared non-target categories with zero raw "
                f"annotations only: {ignored}"
            )

    crowd_splits = {
        split: int(report["crowd_annotations"])
        for split, report in validation.items()
        if int(report["crowd_annotations"]) > 0
    }
    if crowd_splits:
        raise ValueError(
            "AOD-4 primary conversion requires iscrowd=0 for every annotation. "
            "YOLO labels cannot preserve COCO crowd-ignore evaluation semantics: "
            f"{crowd_splits}"
        )
    if len(target_categories) != num_classes:
        raise ValueError(
            f"Expected {num_classes} AOD-4 target categories, found "
            f"{len(target_categories)}: {target_categories}"
        )
    if num_classes != len(AOD4_TARGET_CLASSES):
        raise ValueError(
            f"AOD-4 requires {len(AOD4_TARGET_CLASSES)} target classes, but "
            f"--num_classes={num_classes}"
        )

    actual_official_counts = {
        split: {
            "images": int(validation[split]["images"]),
            "annotations": int(validation[split]["annotations"]),
            "background_images": int(validation[split]["background_images"]),
            "class_annotations": {
                target_categories[category_id]: sum(
                    int(annotation["category_id"]) == category_id
                    for annotation in canonical_by_split[split]["annotations"]
                )
                for category_id in sorted(target_categories)
            },
        }
        for split in SPLITS
    }
    official_count_gate = {
        "enabled": bool(enforce_official_counts),
        "expected": AOD4_V6_OFFICIAL_COUNTS,
        "actual": actual_official_counts,
        "passed": actual_official_counts == AOD4_V6_OFFICIAL_COUNTS,
    }
    if enforce_official_counts and not official_count_gate["passed"]:
        raise ValueError(
            "Input does not match the pinned AOD-4 v6 official export counts: "
            f"actual={actual_official_counts}, expected={AOD4_V6_OFFICIAL_COUNTS}"
        )

    print("[2/5] Decoding and hashing every raw source image")
    decode_report = _verify_image_decoding(raw_coco, data_root, verify_image_decode)
    hash_inventory = build_source_hash_inventory(
        raw_coco,
        data_root,
        cache_path=hash_cache_path,
        annotation_hashes=annotation_hashes,
    )

    if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL:
        print("[2/5] Preserving the published AOD-4 v6 train/val/test membership")
    else:
        print(
            "[2/5] Applying deterministic source ownership "
            f"priority {' > '.join(SOURCE_SPLIT_PRIORITY)}"
        )
    selected_canonical, source_audit = apply_source_split_policy(
        canonical_by_split, hash_inventory, policy=source_split_policy
    )
    print(
        "  Source components: "
        f"raw_cross_split={source_audit['before']['cross_split_source_groups']} "
        f"excluded_images={source_audit['excluded']['images']} "
        f"post_policy_cross_split={source_audit['after']['cross_split_source_groups']}"
    )
    print(
        "  Raw collision signals: "
        + json.dumps(
            source_audit["raw_collision_signals"],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for split in SPLITS:
        before = source_audit["before"]["per_split"][split]
        after = source_audit["after"]["per_split"][split]
        excluded = source_audit["excluded"]["per_split"][split]
        print(
            f"  {split}: raw={before['images']} retained={after['images']} "
            f"excluded={excluded['images']}"
        )
    post_policy_report = _check_post_policy_sources(
        selected_canonical, hash_inventory, source_split_policy
    )
    filtered_by_split = {
        split: filter_annotations(
            selected_canonical[split],
            min_bbox_area,
            min_bbox_side,
            drop_empty_images=False,
        )
        for split in SPLITS
    }
    source_split_counts = {
        split: {
            "images": len(canonical_by_split[split]["images"]),
            "annotations": len(canonical_by_split[split]["annotations"]),
        }
        for split in SPLITS
    }
    filtered_by_split["_data_root"] = data_root
    return {
        "raw_coco": raw_coco,
        "coco_by_split": filtered_by_split,
        "validation": validation,
        "annotation_hashes": annotation_hashes,
        "target_categories": target_categories,
        "source_category_audit": source_category_audit,
        "decode_report": decode_report,
        "source_hash_inventory": hash_inventory,
        "source_split_policy": source_split_policy,
        "source_audit": source_audit,
        "post_policy_source_check": post_policy_report,
        "raw_source_counts": _raw_source_counts(raw_coco, hash_inventory),
        "source_split_counts": source_split_counts,
        "official_count_gate": official_count_gate,
    }


def prepare_source_leakage_controlled_data(*args, **kwargs) -> dict:
    """Compatibility helper for explicitly reproducing the historical split."""
    kwargs["source_split_policy"] = SOURCE_SPLIT_POLICY_EXCLUSIVE
    return prepare_aod4_data(*args, **kwargs)


def _rewrite_yaml_root(yaml_path: str, final_root: str):
    with open(yaml_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["path"] = os.path.abspath(final_root)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _build_yolo_tree(build_dir: str, final_dir: str, coco_by_split: dict,
                     assignments: dict, cat_id_to_label: dict, class_names: list,
                     num_clients: int):
    # Each client receives disjoint local train/val/test partitions.
    for client_id in range(num_clients):
        client_build = os.path.join(build_dir, f"client_{client_id}")
        image_dirs = {}
        for split in SPLITS:
            image_dirs[split] = coco_to_yolo_labels(
                coco_by_split[split],
                os.path.join(coco_by_split["_data_root"], split),
                os.path.join(client_build, split),
                image_ids=assignments[split][client_id],
                cat_id_to_label=cat_id_to_label,
            )
        yaml_path = create_dataset_yaml(
            client_build,
            image_dirs["train"],
            image_dirs["val"],
            image_dirs["test"],
            class_names,
        )
        _rewrite_yaml_root(yaml_path, os.path.join(final_dir, f"client_{client_id}"))

    # Pooled data is the centralized baseline and common/global evaluation set.
    full_build = os.path.join(build_dir, "full")
    full_image_dirs = {}
    for split in SPLITS:
        full_image_dirs[split] = coco_to_yolo_labels(
            coco_by_split[split],
            os.path.join(coco_by_split["_data_root"], split),
            os.path.join(full_build, split),
            image_ids=[int(image["id"]) for image in coco_by_split[split]["images"]],
            cat_id_to_label=cat_id_to_label,
        )
    yaml_path = create_dataset_yaml(
        full_build,
        full_image_dirs["train"],
        full_image_dirs["val"],
        full_image_dirs["test"],
        class_names,
    )
    _rewrite_yaml_root(yaml_path, os.path.join(final_dir, "full"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare reproducible AOD-4 FL partitions")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "data" / "splits"))
    parser.add_argument("--partition", choices=["dirichlet", "iid"], default="dirichlet")
    parser.add_argument("--alpha", dest="dirichlet_alpha", type=float, default=0.4)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_bbox_area", type=float, default=0.0)
    parser.add_argument("--min_bbox_side", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument(
        "--source_split_policy",
        choices=SUPPORTED_SOURCE_SPLIT_POLICIES,
        default=DEFAULT_SOURCE_SPLIT_POLICY,
        help=(
            "official_aod4_v6 preserves the published train/val/test membership; "
            "exclusive_highest_evaluation_priority reproduces the historical "
            "source-exclusive derived split"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--enforce_official_counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the pinned AOD-4 v6 image/annotation/background counts",
    )
    parser.add_argument(
        "--verify_image_hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Required publication audit: hash every raw image and diagnose "
            "exact-byte overlap across official splits"
        ),
    )
    parser.add_argument(
        "--verify_image_decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode every image and require its pixel size to match COCO metadata",
    )
    args = parser.parse_args(argv)

    if args.num_clients < 2:
        parser.error("--num_clients must be >= 2")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    numeric_values = (args.dirichlet_alpha, args.min_bbox_area, args.min_bbox_side)
    if not all(math.isfinite(float(value)) for value in numeric_values):
        parser.error("alpha and bbox thresholds must be finite")
    if args.partition == "dirichlet" and args.dirichlet_alpha <= 0:
        parser.error("--alpha must be > 0")
    if args.min_bbox_area < 0 or args.min_bbox_side < 0:
        parser.error("bbox filtering thresholds must be non-negative")
    if (
        args.source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and (args.min_bbox_area != 0.0 or args.min_bbox_side != 0.0)
    ):
        parser.error(
            "official_aod4_v6 requires --min_bbox_area 0 and --min_bbox_side 0"
        )
    if args.num_classes <= 0:
        parser.error("--num_classes must be positive")
    if not args.verify_image_hashes:
        parser.error(
            "--no-verify_image_hashes is incompatible with publication "
            "source auditing; every source image must be SHA-256 hashed"
        )
    if not args.verify_image_decode:
        parser.error(
            "--no-verify_image_decode is incompatible with publication "
            "preprocessing; every source image must be decoded"
        )

    args.data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(args.data_root):
        parser.error(f"--data_root does not exist: {args.data_root}")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tag = _split_tag(args)
    split_path = os.path.join(output_dir, f"split_{tag}.json")
    yolo_dir = os.path.join(output_dir, f"yolo_{tag}")

    if os.path.exists(split_path) or os.path.exists(yolo_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists for {tag}. Use --overwrite only to replace generated artifacts."
            )
        if os.path.isdir(yolo_dir):
            shutil.rmtree(yolo_dir)
        if os.path.isfile(split_path):
            os.unlink(split_path)

    prepared = prepare_aod4_data(
        args.data_root,
        min_bbox_area=args.min_bbox_area,
        min_bbox_side=args.min_bbox_side,
        num_classes=args.num_classes,
        verify_image_hashes=args.verify_image_hashes,
        verify_image_decode=args.verify_image_decode,
        hash_cache_path=os.path.join(
            output_dir, ".aod4_source_hash_inventory_cache.json"
        ),
        source_split_policy=args.source_split_policy,
        enforce_official_counts=args.enforce_official_counts,
    )
    coco_by_split = prepared["coco_by_split"]
    target_categories = prepared["target_categories"]

    cat_ids = sorted(target_categories)
    cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    class_names = [target_categories[cat_id] for cat_id in cat_ids]
    proportions = draw_client_proportions(
        cat_ids,
        args.num_clients,
        args.partition,
        args.dirichlet_alpha,
        args.seed,
    )

    print("[3/5] Creating source-group-atomic client partitions")
    split_seeds = {"train": args.seed, "val": args.seed + 1001, "test": args.seed + 2001}
    if args.partition == "iid":
        assignments = {
            split: partition_images_iid(
                coco_by_split[split], args.num_clients, split_seeds[split],
                group_atomic=True,
            )
            for split in SPLITS
        }
    else:
        assignments = {
            split: partition_images(
                coco_by_split[split], args.num_clients, proportions, split_seeds[split],
                group_atomic=True,
            )
            for split in SPLITS
        }
    assignments, cross_split_client_report = (
        align_cross_split_source_group_owners(coco_by_split, assignments)
    )
    source_group_counts = {
        split: _client_source_group_counts(coco_by_split[split], assignments[split])
        for split in SPLITS
    }
    partition_validation = {
        split: validate_source_group_assignments(
            coco_by_split[split], assignments[split]
        )
        for split in SPLITS
    }
    stats = {
        split: split_statistics(coco_by_split[split], assignments[split])
        for split in SPLITS
    }
    for split in SPLITS:
        print(f"  {split}:")
        for row in stats[split]:
            print(f"    Client {row['client_id']}: {row}")

    clients = []
    for client_id in range(args.num_clients):
        client = {"client_id": client_id, "splits": {}}
        for split in SPLITS:
            row = stats[split][client_id]
            client["splits"][split] = {
                **row,
                "num_source_groups": source_group_counts[split][client_id],
                "image_ids": assignments[split][client_id],
            }
            client["splits"][split].pop("client_id", None)
        clients.append(client)

    split_counts = {
        split: {
            "images": len(coco_by_split[split]["images"]),
            "annotations": len(coco_by_split[split]["annotations"]),
        }
        for split in SPLITS
    }
    manifest = {
        "metadata": {
            "schema_version": 7,
            "data_root": args.data_root,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
            "partition_algorithm": (
                "random_source_group_lpt_balance_cross_split_owner_v2"
                if args.partition == "iid"
                else "source_group_target_deficit_balance_cross_split_owner_v2"
            ),
            "client_partition_unit": CLIENT_PARTITION_UNIT,
            "num_clients": args.num_clients,
            "seed": args.seed,
            "min_bbox_area": args.min_bbox_area,
            "min_bbox_side": args.min_bbox_side,
            "drop_empty_images": False,
            "crowd_policy": "require_zero_crowd_annotations_for_YOLO_metric_equivalence",
            "category_policy": AOD4_CATEGORY_POLICY,
            "source_category_audit": prepared["source_category_audit"],
            "source_split_policy": prepared["source_split_policy"],
            "official_split_preserved": (
                args.source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
            ),
            "source_identity_policy": SOURCE_IDENTITY_POLICY,
            "source_split_priority": (
                []
                if args.source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
                else list(SOURCE_SPLIT_PRIORITY)
            ),
            "class_names": class_names,
            "cat_id_to_label": {str(key): value for key, value in cat_id_to_label.items()},
            "raw_source_counts": prepared["raw_source_counts"],
            "source_split_counts": prepared["source_split_counts"],
            "official_count_gate": prepared["official_count_gate"],
            "split_counts": split_counts,
            "annotation_sha256": prepared["annotation_hashes"],
            "source_hash_inventory": prepared["source_hash_inventory"],
            "cross_split_source_audit": prepared["source_audit"],
            "post_policy_cross_split_source_check": prepared[
                "post_policy_source_check"
            ],
            "image_hash_check_enabled": True,
            "image_decode_check": prepared["decode_report"],
            "client_target_proportions": proportions,
            "cross_split_client_source_group_check": cross_split_client_report,
            "realized_partition_statistics": stats,
            "quantity_balance": {
                split: {
                    "min_images": min(row["num_images"] for row in stats[split]),
                    "max_images": max(row["num_images"] for row in stats[split]),
                    "max_min_gap": (
                        max(row["num_images"] for row in stats[split])
                        - min(row["num_images"] for row in stats[split])
                    ),
                    "min_source_groups": min(source_group_counts[split]),
                    "max_source_groups": max(source_group_counts[split]),
                    "largest_source_group_images": partition_validation[split][
                        "largest_source_group_images"
                    ],
                    "balance_bound_images": partition_validation[split][
                        "largest_source_group_images"
                    ],
                    "balance_bound_satisfied": (
                        partition_validation[split]["max_min_image_gap"]
                        <= partition_validation[split][
                            "largest_source_group_images"
                        ]
                    ),
                    "source_group_atomic": True,
                }
                for split in SPLITS
            },
            "generated_yolo_tree_sha256": None,
        },
        "clients": clients,
    }

    print("[4/5] Building YOLO directory atomically")
    build_dir = tempfile.mkdtemp(prefix=f".build_yolo_{tag}_", dir=output_dir)
    try:
        _build_yolo_tree(
            build_dir,
            yolo_dir,
            coco_by_split,
            assignments,
            cat_id_to_label,
            class_names,
            args.num_clients,
        )
        os.replace(build_dir, yolo_dir)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    manifest["metadata"]["generated_yolo_tree_sha256"] = (
        generated_yolo_tree_sha256(yolo_dir)
    )

    print("[5/5] Writing immutable split manifest")
    temp_manifest = split_path + ".tmp"
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(temp_manifest, split_path)
    print(f"Split manifest: {split_path}")
    print(f"YOLO data:     {yolo_dir}")


if __name__ == "__main__":
    main()
