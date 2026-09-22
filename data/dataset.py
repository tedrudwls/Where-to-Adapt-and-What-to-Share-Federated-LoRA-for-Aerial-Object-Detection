"""AOD-4 COCO validation, image-atomic partitioning and YOLO conversion.

The primary protocol retains every valid small-object annotation and every
background image. Client splits are disjoint, source-group atomic and use a
multi-object-aware, balance-constrained assignment toward Dirichlet targets.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


SPLITS = ("train", "val", "test")
BACKGROUND_KEY = "__background__"
AOD4_TARGET_CLASSES = ("airplane", "bird", "drone", "helicopter")
AOD4_V6_OFFICIAL_COUNTS = {
    "train": {
        "images": 15761,
        "annotations": 22058,
        "background_images": 415,
        "class_annotations": {
            "airplane": 5508, "bird": 5522, "drone": 5500, "helicopter": 5528,
        },
    },
    "val": {
        "images": 4514,
        "annotations": 6369,
        "background_images": 125,
        "class_annotations": {
            "airplane": 1625, "bird": 1557, "drone": 1602, "helicopter": 1585,
        },
    },
    "test": {
        "images": 2241,
        "annotations": 3171,
        "background_images": 56,
        "class_annotations": {
            "airplane": 767, "bird": 821, "drone": 796, "helicopter": 787,
        },
    },
}
AOD4_CATEGORY_POLICY = (
    "exact_aod4_targets_ignore_only_unreferenced_declared_categories"
)
SOURCE_SPLIT_POLICY_OFFICIAL = "official_aod4_v6"
SOURCE_SPLIT_POLICY_EXCLUSIVE = "exclusive_highest_evaluation_priority"
DEFAULT_SOURCE_SPLIT_POLICY = SOURCE_SPLIT_POLICY_OFFICIAL
SUPPORTED_SOURCE_SPLIT_POLICIES = (
    SOURCE_SPLIT_POLICY_OFFICIAL,
    SOURCE_SPLIT_POLICY_EXCLUSIVE,
)
SOURCE_IDENTITY_POLICY = "roboflow_source_key_or_exact_sha256_connected_components"
SOURCE_SPLIT_PRIORITY = ("test", "val", "train")
SOURCE_GROUP_FIELD = "_source_group_id"
CLIENT_PARTITION_UNIT = "source_group"
_ROBOFLOW_EXPORT_SUFFIX = re.compile(
    r"\.rf\.[0-9a-f]{20,64}(?=\.[^./\\]+$)",
    flags=re.IGNORECASE,
)


def source_image_key(file_name: str) -> str:
    """Return a conservative, case-normalized pre-Roboflow filename key."""
    normalized = str(file_name).replace("\\", "/").strip().lower()
    return _ROBOFLOW_EXPORT_SUFFIX.sub("", normalized)


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_coco_annotations(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"{json_path}: missing COCO key '{key}'")
    return data


def annotation_sha256(json_path: str) -> str:
    return file_sha256(json_path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_stat_records(coco_by_split: dict, data_root: str) -> dict:
    records = {}
    for split in SPLITS:
        split_records = []
        for image in coco_by_split[split]["images"]:
            image_id = int(image["id"])
            relative_name = str(image["file_name"]).replace("\\", "/")
            path = os.path.join(data_root, split, *relative_name.split("/"))
            stat = os.stat(path)
            split_records.append([
                image_id,
                relative_name,
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            ])
        records[split] = sorted(split_records, key=lambda row: (row[1], row[0]))
    return records


def _tree_digest_from_inventory_records(records: List[list]) -> str:
    digest = hashlib.sha256()
    for _, relative_name, content_digest in sorted(
        records, key=lambda row: (str(row[1]), int(row[0]))
    ):
        digest.update(str(relative_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_digest).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source_hash_inventory(coco_by_split: dict, inventory: dict) -> dict:
    """Validate a manifest inventory and return ``split -> image_id -> record``.

    Records are compact JSON triples ``[image_id, relative_filename, sha256]``.
    This function performs no file reads; callers first verify the raw image-tree
    digest against disk, then use these per-file hashes to deterministically
    reproduce source connected components.
    """
    if not isinstance(inventory, dict) or int(inventory.get("schema_version", -1)) != 1:
        raise ValueError("Source hash inventory has an unsupported schema")
    if inventory.get("identity_policy") != SOURCE_IDENTITY_POLICY:
        raise ValueError("Source hash inventory identity policy is stale")
    raw_records = inventory.get("records")
    tree_digests = inventory.get("per_split_image_tree_sha256")
    if not isinstance(raw_records, dict) or set(raw_records) != set(SPLITS):
        raise ValueError("Source hash inventory records must cover train/val/test")
    if not isinstance(tree_digests, dict) or set(tree_digests) != set(SPLITS):
        raise ValueError("Source hash inventory tree digests must cover train/val/test")

    maps = {}
    for split in SPLITS:
        expected = {
            int(image["id"]): str(image["file_name"]).replace("\\", "/")
            for image in coco_by_split[split]["images"]
        }
        if len(expected) != len(coco_by_split[split]["images"]):
            raise ValueError(f"{split}: duplicate image id while validating hash inventory")
        split_map = {}
        normalized_records = []
        for row in raw_records[split]:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{split}: invalid source hash inventory record {row!r}")
            image_id, relative_name, digest = int(row[0]), str(row[1]), str(row[2]).lower()
            if image_id in split_map:
                raise ValueError(f"{split}: duplicate inventory image id {image_id}")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{split}: invalid SHA-256 for image {image_id}")
            split_map[image_id] = {
                "image_id": image_id,
                "file_name": relative_name,
                "sha256": digest,
            }
            normalized_records.append([image_id, relative_name, digest])
        if set(split_map) != set(expected):
            raise ValueError(
                f"{split}: source hash inventory image IDs differ from COCO"
            )
        mismatched_names = [
            image_id for image_id, relative_name in expected.items()
            if split_map[image_id]["file_name"] != relative_name
        ]
        if mismatched_names:
            raise ValueError(
                f"{split}: source hash inventory filenames differ for IDs "
                f"{mismatched_names[:10]}"
            )
        actual_tree_digest = _tree_digest_from_inventory_records(normalized_records)
        if actual_tree_digest != str(tree_digests[split]):
            raise ValueError(f"{split}: source hash inventory tree digest is inconsistent")
        maps[split] = split_map

    digest_payload = {
        key: inventory[key]
        for key in (
            "schema_version", "identity_policy", "records",
            "per_split_image_tree_sha256",
        )
    }
    if _json_sha256(digest_payload) != inventory.get("inventory_sha256"):
        raise ValueError("Source hash inventory digest is inconsistent")
    return maps


def build_source_hash_inventory(
    coco_by_split: dict,
    data_root: str,
    *,
    cache_path: Optional[str] = None,
    annotation_hashes: Optional[dict] = None,
    force_rehash: bool = False,
) -> dict:
    """Hash source files and return a compact reproducible per-file inventory.

    When ``cache_path`` is supplied, content hashes are reused only when the
    absolute data root, annotation fingerprints, complete file set, and every
    file's size/mtime/ctime are unchanged. Publication manifests still embed
    the returned per-file hashes and raw tree digests; the cache is only an I/O
    optimization for generating multiple alpha/seed partitions.
    """
    data_root = os.path.abspath(data_root)
    source_annotation_hashes = (
        {split: str(annotation_hashes[split]) for split in SPLITS}
        if annotation_hashes is not None
        else {split: _json_sha256(coco_by_split[split]) for split in SPLITS}
    )
    stat_records = _source_file_stat_records(coco_by_split, data_root)
    cache_key = {
        "data_root": data_root,
        "annotation_sha256": source_annotation_hashes,
        "file_stat_sha256": _json_sha256(stat_records),
    }
    if cache_path and not force_rehash and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                isinstance(cached, dict)
                and int(cached.get("cache_schema_version", -1)) == 1
                and cached.get("cache_key") == cache_key
            ):
                inventory = cached.get("inventory")
                validate_source_hash_inventory(coco_by_split, inventory)
                print(f"[Data] Reusing source SHA-256 inventory cache: {cache_path}")
                return copy.deepcopy(inventory)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    records = {}
    processed = 0
    total = sum(len(coco_by_split[split]["images"]) for split in SPLITS)
    for split in SPLITS:
        split_records = []
        for image in coco_by_split[split]["images"]:
            image_id = int(image["id"])
            relative_name = str(image["file_name"]).replace("\\", "/")
            path = os.path.join(data_root, split, *relative_name.split("/"))
            split_records.append([image_id, relative_name, file_sha256(path)])
            processed += 1
            if processed % 2000 == 0:
                print(f"  Hashed {processed}/{total} source images...")
        records[split] = sorted(split_records, key=lambda row: (row[1], row[0]))

    inventory = {
        "schema_version": 1,
        "identity_policy": SOURCE_IDENTITY_POLICY,
        "records": records,
        "per_split_image_tree_sha256": {
            split: _tree_digest_from_inventory_records(records[split])
            for split in SPLITS
        },
    }
    inventory["inventory_sha256"] = _json_sha256(inventory)
    validate_source_hash_inventory(coco_by_split, inventory)

    if cache_path:
        cache_path = os.path.abspath(cache_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "cache_schema_version": 1,
            "cache_key": cache_key,
            "inventory": inventory,
        }
        temporary = f"{cache_path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    return inventory


def image_tree_sha256(coco: dict, image_dir: str) -> str:
    """Digest the sorted (relative filename, file-content SHA-256) image tree."""
    digest = hashlib.sha256()
    records = sorted(
        (str(image["file_name"]).replace("\\", "/"), image)
        for image in coco["images"]
    )
    for relative_name, _ in records:
        path = os.path.join(image_dir, *relative_name.split("/"))
        content_digest = file_sha256(path)
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def image_tree_stat_sha256(coco: dict, image_dir: str) -> str:
    """Fast cache key for filenames and size/mtime/ctime metadata."""
    digest = hashlib.sha256()
    relative_names = sorted(
        str(image["file_name"]).replace("\\", "/") for image in coco["images"]
    )
    for relative_name in relative_names:
        path = os.path.join(image_dir, *relative_name.split("/"))
        stat = os.stat(path)
        record = (
            relative_name,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_selection_stats(
    coco: dict,
    image_ids: Iterable[int],
    group_by_image_id: Dict[int, str],
) -> dict:
    selected = {int(value) for value in image_ids}
    annotations = [
        annotation for annotation in coco["annotations"]
        if int(annotation["image_id"]) in selected
    ]
    annotated_ids = {int(annotation["image_id"]) for annotation in annotations}
    class_instances = defaultdict(int)
    class_images = defaultdict(set)
    for annotation in annotations:
        category_id = str(int(annotation["category_id"]))
        class_instances[category_id] += 1
        class_images[category_id].add(int(annotation["image_id"]))
    category_ids = [str(value) for value in sorted(category_mapping(coco))]
    return {
        "images": len(selected),
        "annotations": len(annotations),
        "background_images": len(selected - annotated_ids),
        "source_groups": len({group_by_image_id[image_id] for image_id in selected}),
        "class_instances": {
            category_id: int(class_instances[category_id]) for category_id in category_ids
        },
        "class_images": {
            category_id: len(class_images[category_id]) for category_id in category_ids
        },
    }


def apply_source_split_policy(
    coco_by_split: dict,
    hash_inventory: dict,
    policy: str = DEFAULT_SOURCE_SPLIT_POLICY,
) -> Tuple[dict, dict]:
    """Annotate source components and apply the selected official-split policy.

    Images are connected when they share either a Roboflow-suffix-stripped
    source filename or exact SHA-256 bytes. ``official_aod4_v6`` preserves every
    image in its published train/val/test directory. The optional historical
    ``exclusive_highest_evaluation_priority`` policy retains a component only
    in test, then validation, then train. In both modes source-group IDs are
    attached to copied image records for group-atomic FL client partitioning.
    The input COCO mappings are never mutated.
    """
    if policy not in SUPPORTED_SOURCE_SPLIT_POLICIES:
        raise ValueError(
            f"Unsupported source split policy {policy!r}; expected one of "
            f"{SUPPORTED_SOURCE_SPLIT_POLICIES}"
        )
    inventory_maps = validate_source_hash_inventory(coco_by_split, hash_inventory)
    priority_rank = {split: index for index, split in enumerate(SOURCE_SPLIT_PRIORITY)}
    if set(priority_rank) != set(SPLITS):
        raise RuntimeError("Source split priority must contain train/val/test exactly once")

    records = []
    for split in SPLITS:
        for image in coco_by_split[split]["images"]:
            image_id = int(image["id"])
            item = inventory_maps[split][image_id]
            records.append({
                "key": (split, image_id),
                "split": split,
                "image_id": image_id,
                "file_name": item["file_name"],
                "source_key": source_image_key(item["file_name"]),
                "sha256": item["sha256"],
            })
    records.sort(key=lambda row: (row["split"], row["file_name"], row["image_id"]))

    parent = {record["key"]: record["key"] for record in records}

    def find(key):
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # Lexicographic root selection makes the component independent of input order.
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    source_buckets = defaultdict(list)
    hash_buckets = defaultdict(list)
    exact_name_buckets = defaultdict(list)
    for record in records:
        source_buckets[record["source_key"]].append(record)
        hash_buckets[record["sha256"]].append(record)
        exact_name_buckets[record["file_name"].lower()].append(record)
    for buckets in (source_buckets, hash_buckets):
        for members in buckets.values():
            anchor = members[0]["key"]
            for member in members[1:]:
                union(anchor, member["key"])

    components = defaultdict(list)
    for record in records:
        components[find(record["key"])].append(record)

    component_info = {}
    group_by_record = {}
    retained_ids = {split: set() for split in SPLITS}
    excluded_ids = {split: set() for split in SPLITS}
    cross_split_components = []
    for members in components.values():
        members = sorted(
            members,
            key=lambda row: (row["split"], row["file_name"], row["image_id"]),
        )
        component_payload = [
            [
                member["split"], member["image_id"], member["file_name"],
                member["source_key"], member["sha256"],
            ]
            for member in members
        ]
        group_id = _json_sha256(component_payload)
        member_splits = sorted(
            {member["split"] for member in members}, key=priority_rank.get
        )
        owner_split = min(member_splits, key=priority_rank.get)
        info = {
            "group_id": group_id,
            "owner_split": owner_split,
            "member_splits": member_splits,
            "members": members,
        }
        component_info[group_id] = info
        if len(member_splits) > 1:
            cross_split_components.append(info)
        for member in members:
            group_by_record[member["key"]] = group_id
            retain = (
                policy == SOURCE_SPLIT_POLICY_OFFICIAL
                or member["split"] == owner_split
            )
            target = retained_ids if retain else excluded_ids
            target[member["split"]].add(member["image_id"])

    cleaned = {}
    before_group_maps = {}
    after_group_maps = {}
    for split in SPLITS:
        before_group_maps[split] = {
            int(image["id"]): group_by_record[(split, int(image["id"]))]
            for image in coco_by_split[split]["images"]
        }
        after_group_maps[split] = {
            image_id: before_group_maps[split][image_id]
            for image_id in retained_ids[split]
        }
        selected = copy.deepcopy(coco_by_split[split])
        selected_images = []
        for image in selected["images"]:
            image_id = int(image["id"])
            if image_id in retained_ids[split]:
                image[SOURCE_GROUP_FIELD] = after_group_maps[split][image_id]
                selected_images.append(image)
        selected["images"] = selected_images
        selected["annotations"] = [
            annotation for annotation in selected["annotations"]
            if int(annotation["image_id"]) in retained_ids[split]
        ]
        cleaned[split] = selected

    cross_source_keys = [
        key for key, members in source_buckets.items()
        if len({member["split"] for member in members}) > 1
    ]
    cross_hashes = [
        digest for digest, members in hash_buckets.items()
        if len({member["split"] for member in members}) > 1
    ]
    cross_exact_names = [
        name for name, members in exact_name_buckets.items()
        if len({member["split"] for member in members}) > 1
    ]
    membership_rows = sorted(
        [
            record["split"], record["image_id"], record["file_name"],
            group_by_record[record["key"]],
            component_info[group_by_record[record["key"]]]["owner_split"],
        ]
        for record in records
    )
    retained_rows = sorted(
        [split, image_id, before_group_maps[split][image_id]]
        for split in SPLITS for image_id in retained_ids[split]
    )
    excluded_rows = sorted(
        [split, image_id, before_group_maps[split][image_id]]
        for split in SPLITS for image_id in excluded_ids[split]
    )
    examples = []
    for info in sorted(cross_split_components, key=lambda value: value["group_id"])[:10]:
        examples.append({
            "source_group_id": info["group_id"],
            "owner_split": info["owner_split"],
            "member_counts": {
                split: sum(member["split"] == split for member in info["members"])
                for split in SPLITS
            },
            "members": [
                [member["split"], member["image_id"], member["file_name"]]
                for member in info["members"][:10]
            ],
        })

    before_stats = {
        split: _source_selection_stats(
            coco_by_split[split],
            before_group_maps[split],
            before_group_maps[split],
        )
        for split in SPLITS
    }
    after_stats = {
        split: _source_selection_stats(
            coco_by_split[split],
            retained_ids[split],
            before_group_maps[split],
        )
        for split in SPLITS
    }
    excluded_stats = {
        split: _source_selection_stats(
            coco_by_split[split],
            excluded_ids[split],
            before_group_maps[split],
        )
        for split in SPLITS
    }
    audit = {
        "schema_version": 1,
        "policy": policy,
        "identity_policy": SOURCE_IDENTITY_POLICY,
        "split_priority": (
            [] if policy == SOURCE_SPLIT_POLICY_OFFICIAL
            else list(SOURCE_SPLIT_PRIORITY)
        ),
        "before": {
            "per_split": before_stats,
            "unique_source_groups": len(components),
            "cross_split_source_groups": len(cross_split_components),
        },
        "after": {
            "per_split": after_stats,
            "unique_source_groups": len(components),
            "cross_split_source_groups": (
                len(cross_split_components)
                if policy == SOURCE_SPLIT_POLICY_OFFICIAL else 0
            ),
        },
        "excluded": {
            "per_split": excluded_stats,
            "images": sum(len(values) for values in excluded_ids.values()),
            "source_groups": len({row[2] for row in excluded_rows}),
        },
        "raw_collision_signals": {
            "exact_filename_cross_split_keys": len(cross_exact_names),
            "roboflow_source_key_cross_split_keys": len(cross_source_keys),
            "exact_sha256_cross_split_hashes": len(cross_hashes),
        },
        "digests": {
            "component_membership_sha256": _json_sha256(membership_rows),
            "retained_selection_sha256": _json_sha256(retained_rows),
            "excluded_selection_sha256": _json_sha256(excluded_rows),
        },
        "examples": examples,
        "per_split_image_tree_sha256": copy.deepcopy(
            hash_inventory["per_split_image_tree_sha256"]
        ),
    }
    audit["audit_sha256"] = _json_sha256(audit)
    return cleaned, audit


def apply_source_leakage_policy(
    coco_by_split: dict,
    hash_inventory: dict,
) -> Tuple[dict, dict]:
    """Compatibility wrapper for the historical source-exclusive policy."""
    return apply_source_split_policy(
        coco_by_split,
        hash_inventory,
        policy=SOURCE_SPLIT_POLICY_EXCLUSIVE,
    )


def generated_yolo_tree_sha256(yolo_dir: str) -> str:
    """Hash generated YAML/labels and image-symlink targets, excluding runtime caches."""
    if not os.path.isdir(yolo_dir):
        raise FileNotFoundError(f"Generated YOLO directory not found: {yolo_dir}")
    records = []
    for directory, _, filenames in os.walk(yolo_dir):
        for filename in filenames:
            if filename.endswith(".cache"):
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, yolo_dir).replace(os.sep, "/")
            if os.path.islink(path):
                records.append((relative, "symlink", os.path.realpath(path)))
            else:
                records.append((relative, "file", file_sha256(path)))
    if not records:
        raise ValueError(f"Generated YOLO tree is empty: {yolo_dir}")
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_source_image_trees(args, metadata: dict) -> str:
    """Cryptographically verify once, then reuse only while every file stat is unchanged."""
    manifest_digest = annotation_sha256(args.split_file)
    source_audit_metadata = metadata.get("cross_split_source_audit", {})
    source_inventory = metadata.get("source_hash_inventory")
    expected_digests = (
        source_inventory.get("per_split_image_tree_sha256")
        if isinstance(source_inventory, dict)
        else source_audit_metadata.get("per_split_image_tree_sha256")
    )
    if not metadata.get("image_hash_check_enabled"):
        print(
            "[WARN] Split was generated without image-content hashes; source image "
            "immutability cannot be verified for this run"
        )
        return manifest_digest
    if not isinstance(expected_digests, dict):
        raise ValueError(
            "Hash-enabled manifest has no per-split image-tree digest; regenerate the split"
        )

    coco_by_split = {
        split: load_coco_annotations(
            os.path.join(args.data_root, split, "_annotations.coco.json")
        )
        for split in SPLITS
    }
    if isinstance(source_inventory, dict):
        validate_source_hash_inventory(coco_by_split, source_inventory)
        audit_digests = source_audit_metadata.get("per_split_image_tree_sha256")
        if audit_digests is not None and audit_digests != expected_digests:
            raise ValueError(
                "Source audit and source hash inventory contain different tree digests"
            )
    stat_digests = {
        split: image_tree_stat_sha256(coco_by_split[split], os.path.join(args.data_root, split))
        for split in SPLITS
    }
    cache_path = args.split_file + ".image_verify_cache.json"
    cache = None
    if not getattr(args, "rehash_source_images", False) and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, ValueError):
            cache = None
    cache_valid = bool(
        isinstance(cache, dict)
        and cache.get("schema_version") == 1
        and cache.get("split_manifest_sha256") == manifest_digest
        and cache.get("stat_sha256") == stat_digests
        and cache.get("content_sha256") == expected_digests
    )
    if cache_valid:
        print("[Data] Source image SHA-256 verification cache is valid")
        return manifest_digest

    print("[Data] Hashing source image trees (first run or cache invalidated)...")
    actual_digests = {
        split: image_tree_sha256(coco_by_split[split], os.path.join(args.data_root, split))
        for split in SPLITS
    }
    mismatches = [
        split for split in SPLITS if actual_digests[split] != expected_digests.get(split)
    ]
    if mismatches:
        raise ValueError(
            "Source image bytes or filenames changed after split generation for: "
            + ", ".join(mismatches)
        )
    payload = {
        "schema_version": 1,
        "split_manifest_sha256": manifest_digest,
        "stat_sha256": stat_digests,
        "content_sha256": actual_digests,
    }
    temporary = f"{cache_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)
    return manifest_digest


def category_mapping(coco: dict) -> Dict[int, str]:
    mapping = {}
    for category in coco["categories"]:
        cid = int(category["id"])
        name = str(category["name"]).strip()
        if cid in mapping:
            raise ValueError(f"Duplicate category id {cid}")
        if not name:
            raise ValueError(f"Empty category name for id {cid}")
        mapping[cid] = name
    return mapping


def _normalized_category_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def canonicalize_aod4_categories(coco: dict) -> Tuple[dict, dict]:
    """Project COCO metadata to the four AOD-4 targets without dropping GT.

    Some Roboflow AOD-4 exports declare an additional aggregate category such
    as ``airplane-helicopter-drone-bird`` even though no annotation references
    it.  A declared non-target category is safe to omit from the model label
    space only when its *raw* annotation count is exactly zero.  Referenced
    non-target categories fail fast instead of silently turning objects into
    background or changing this study into a five-class task.

    The returned COCO mapping shares the input images/annotations (which are
    treated as read-only here) and contains only the selected category records.
    The JSON-serializable audit is stored in the immutable split manifest and
    recomputed by preflight/runtime validation.
    """
    declared = category_mapping(coco)
    expected_by_normalized = {
        _normalized_category_name(name): name for name in AOD4_TARGET_CLASSES
    }
    if len(expected_by_normalized) != len(AOD4_TARGET_CLASSES):
        raise RuntimeError("AOD-4 target class normalization is ambiguous")

    matches = {key: [] for key in expected_by_normalized}
    for category_id, name in declared.items():
        normalized = _normalized_category_name(name)
        if normalized in matches:
            matches[normalized].append(category_id)

    missing = [
        expected_by_normalized[key] for key, category_ids in matches.items()
        if not category_ids
    ]
    duplicate = {
        expected_by_normalized[key]: category_ids
        for key, category_ids in matches.items()
        if len(category_ids) > 1
    }
    if missing or duplicate:
        raise ValueError(
            "COCO categories do not resolve to exactly one id for every AOD-4 "
            f"target; missing={missing}, duplicate={duplicate}, declared={declared}"
        )

    target_ids = {
        category_ids[0] for category_ids in matches.values()
    }
    raw_counts = {category_id: 0 for category_id in declared}
    for annotation in coco["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in raw_counts:
            raise ValueError(
                f"Annotation references undeclared category id {category_id}"
            )
        raw_counts[category_id] += 1

    referenced_non_targets = {
        category_id: {
            "name": declared[category_id],
            "raw_annotations": raw_counts[category_id],
        }
        for category_id in sorted(set(declared) - target_ids)
        if raw_counts[category_id] > 0
    }
    if referenced_non_targets:
        raise ValueError(
            "Non-target COCO categories contain ground-truth annotations and "
            "cannot be silently ignored or merged. Correct/remap the source "
            f"labels before training: {referenced_non_targets}"
        )

    selected_records = [
        copy.deepcopy(category)
        for category in coco["categories"]
        if int(category["id"]) in target_ids
    ]
    canonical = dict(coco)
    canonical["categories"] = selected_records
    selected = category_mapping(canonical)
    ignored = {
        str(category_id): declared[category_id]
        for category_id in sorted(set(declared) - target_ids)
    }
    audit = {
        "declared_categories": {
            str(category_id): declared[category_id] for category_id in sorted(declared)
        },
        "target_categories": {
            str(category_id): selected[category_id] for category_id in sorted(selected)
        },
        "raw_annotation_counts": {
            str(category_id): int(raw_counts[category_id])
            for category_id in sorted(raw_counts)
        },
        "ignored_unreferenced_categories": ignored,
    }
    return canonical, audit


def validate_coco(coco: dict, image_dir: str, split_name: str,
                  expected_categories: Optional[Dict[int, str]] = None) -> dict:
    """Fail fast on corrupt IDs/files and return validation statistics."""
    categories = category_mapping(coco)
    if expected_categories is not None and categories != expected_categories:
        raise ValueError(
            f"{split_name}: category id/name mapping differs from train: "
            f"{categories} != {expected_categories}"
        )

    image_by_id = {}
    file_names = set()
    label_stems = set()
    missing_files = []
    for image in coco["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError(f"{split_name}: duplicate image id {image_id}")
        width, height = int(image.get("width", 0)), int(image.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"{split_name}: image {image_id} has invalid size {width}x{height}")
        filename = str(image["file_name"])
        if os.path.isabs(filename) or ".." in filename.replace("\\", "/").split("/"):
            raise ValueError(f"{split_name}: unsafe file_name {filename!r}")
        if filename in file_names:
            raise ValueError(f"{split_name}: duplicate file_name {filename!r}")
        file_names.add(filename)
        label_stem = os.path.splitext(filename)[0]
        if label_stem in label_stems:
            raise ValueError(
                f"{split_name}: image filenames collide after extension removal: {label_stem!r}"
            )
        label_stems.add(label_stem)
        image_by_id[image_id] = image
        if not os.path.isfile(os.path.join(image_dir, filename)):
            missing_files.append(filename)

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{split_name}: {len(missing_files)} referenced images are missing. First entries:\n{preview}"
        )

    annotation_ids = set()
    invalid_boxes = 0
    crowd_count = 0
    for ann in coco["annotations"]:
        ann_id = int(ann["id"])
        if ann_id in annotation_ids:
            raise ValueError(f"{split_name}: duplicate annotation id {ann_id}")
        annotation_ids.add(ann_id)
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in image_by_id:
            raise ValueError(f"{split_name}: annotation {ann_id} references unknown image {image_id}")
        if category_id not in categories:
            raise ValueError(f"{split_name}: annotation {ann_id} references unknown category {category_id}")
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or not all(np.isfinite(float(v)) for v in bbox):
            invalid_boxes += 1
            continue
        x, y, width, height = map(float, bbox)
        if width <= 0 or height <= 0:
            invalid_boxes += 1
        else:
            image = image_by_id[image_id]
            intersection_width = min(float(image["width"]), x + width) - max(0.0, x)
            intersection_height = min(float(image["height"]), y + height) - max(0.0, y)
            if intersection_width <= 0 or intersection_height <= 0:
                invalid_boxes += 1
        crowd_count += int(bool(ann.get("iscrowd", 0)))
    if invalid_boxes:
        raise ValueError(f"{split_name}: {invalid_boxes} annotations have invalid bounding boxes")

    annotated_ids = {int(a["image_id"]) for a in coco["annotations"]}
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "background_images": len(coco["images"]) - len(annotated_ids),
        "crowd_annotations": crowd_count,
        "categories": categories,
    }


def filter_annotations(coco: dict, min_bbox_area: float = 0.0,
                       min_bbox_side: float = 0.0,
                       drop_empty_images: bool = False) -> dict:
    """Optional annotation-quality ablation; primary experiments use zeros."""
    if (
        not math.isfinite(float(min_bbox_area))
        or not math.isfinite(float(min_bbox_side))
        or min_bbox_area < 0
        or min_bbox_side < 0
    ):
        raise ValueError("Bounding-box thresholds must be finite and non-negative")
    result = copy.deepcopy(coco)
    image_by_id = {int(image["id"]): image for image in result["images"]}
    kept = []
    for ann in result["annotations"]:
        image = image_by_id[int(ann["image_id"])]
        clipped = _clip_bbox_xywh(
            ann["bbox"], int(image["width"]), int(image["height"])
        )
        if clipped is None:
            continue
        width = clipped[2] * int(image["width"])
        height = clipped[3] * int(image["height"])
        if (
            width * height >= min_bbox_area
            and width >= min_bbox_side
            and height >= min_bbox_side
        ):
            kept.append(ann)
    result["annotations"] = kept
    if drop_empty_images:
        nonempty = {int(ann["image_id"]) for ann in kept}
        result["images"] = [img for img in result["images"] if int(img["id"]) in nonempty]
    return result


def build_image_ann_map(coco: dict) -> Dict[int, List[dict]]:
    mapping = defaultdict(list)
    for ann in coco["annotations"]:
        mapping[int(ann["image_id"])].append(ann)
    return dict(mapping)


def _image_histograms(coco: dict, cat_ids: List[int]) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    cat_index = {cid: idx for idx, cid in enumerate(cat_ids)}
    anns_by_image = build_image_ann_map(coco)
    histograms = {}
    totals = np.zeros(len(cat_ids) + 1, dtype=np.float64)
    for image in coco["images"]:
        image_id = int(image["id"])
        hist = np.zeros(len(cat_ids) + 1, dtype=np.float64)
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0):
                continue
            hist[cat_index[int(ann["category_id"])]] += 1.0
        if hist[:-1].sum() == 0:
            hist[-1] = 1.0
        histograms[image_id] = hist
        totals += hist
    return histograms, totals


def draw_client_proportions(cat_ids: List[int], num_clients: int, partition: str,
                            alpha: float, seed: int) -> Dict[str, List[float]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if partition not in ("iid", "dirichlet"):
        raise ValueError(f"Unknown partition mode: {partition!r}")
    if partition == "dirichlet" and (not math.isfinite(float(alpha)) or alpha <= 0):
        raise ValueError("Dirichlet alpha must be finite and positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    keys = [str(cid) for cid in cat_ids] + [BACKGROUND_KEY]
    result = {}
    for key in keys:
        if partition == "iid" or key == BACKGROUND_KEY:
            values = np.full(num_clients, 1.0 / num_clients)
        else:
            values = rng.dirichlet(np.full(num_clients, alpha))
        result[key] = [float(value) for value in values]
    return result


def _source_groups(coco: dict) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    seen_ids = set()
    for image in coco["images"]:
        image_id = int(image["id"])
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image id {image_id} in source-group partition")
        seen_ids.add(image_id)
        group_id = image.get(SOURCE_GROUP_FIELD)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                f"Image {image_id} has no valid {SOURCE_GROUP_FIELD}; apply the "
                "source leakage policy before group-atomic partitioning"
            )
        groups[group_id].append(image_id)
    return {
        group_id: sorted(image_ids)
        for group_id, image_ids in sorted(groups.items())
    }


def validate_source_group_assignments(coco: dict, assignments: List[List[int]]) -> dict:
    """Require a nonempty, disjoint cover with each source group in one client."""
    groups = _source_groups(coco)
    expected_ids = {image_id for image_ids in groups.values() for image_id in image_ids}
    image_to_group = {
        image_id: group_id for group_id, image_ids in groups.items()
        for image_id in image_ids
    }
    flat = [int(image_id) for client in assignments for image_id in client]
    if any(len(client) == 0 for client in assignments):
        raise ValueError("Every client must receive at least one source group")
    if len(flat) != len(set(flat)) or set(flat) != expected_ids:
        raise ValueError("Group-atomic assignments are not a disjoint image cover")
    group_owner = {}
    for client_id, image_ids in enumerate(assignments):
        for image_id in image_ids:
            group_id = image_to_group[int(image_id)]
            previous = group_owner.setdefault(group_id, client_id)
            if previous != client_id:
                raise ValueError(
                    f"Source group {group_id} is split across clients {previous}/{client_id}"
                )
    sizes = [len(client) for client in assignments]
    group_counts = [
        len({image_to_group[int(image_id)] for image_id in client})
        for client in assignments
    ]
    return {
        "client_image_counts": sizes,
        "client_source_group_counts": group_counts,
        "max_min_image_gap": max(sizes) - min(sizes),
        "largest_source_group_images": max(map(len, groups.values())),
    }


def align_cross_split_source_group_owners(
    coco_by_split: dict,
    assignments_by_split: Dict[str, List[List[int]]],
) -> Tuple[Dict[str, List[List[int]]], dict]:
    """Keep an audited source component on one synthetic client across splits.

    Official AOD-4 membership is unchanged. Shared components take the client
    owner from train, then validation, then test. Whole non-shared groups are
    moved deterministically afterward only when needed to restore the usual
    indivisible-group quantity bound.
    """
    if set(assignments_by_split) != set(SPLITS):
        raise ValueError("Cross-split alignment requires train/val/test assignments")
    num_clients = len(assignments_by_split["train"])
    if num_clients <= 0 or any(
        len(assignments_by_split[split]) != num_clients for split in SPLITS
    ):
        raise ValueError("All splits must contain the same positive client count")

    groups_by_split = {split: _source_groups(coco_by_split[split]) for split in SPLITS}
    aligned = {
        split: [list(map(int, values)) for values in assignments_by_split[split]]
        for split in SPLITS
    }

    def owners_for(split: str) -> dict:
        image_to_group = {
            image_id: group_id
            for group_id, image_ids in groups_by_split[split].items()
            for image_id in image_ids
        }
        owners = {}
        for client_id, image_ids in enumerate(aligned[split]):
            for image_id in image_ids:
                group_id = image_to_group[int(image_id)]
                previous = owners.setdefault(group_id, client_id)
                if previous != client_id:
                    raise ValueError(
                        f"Source group {group_id} crosses clients in {split}"
                    )
        return owners

    group_splits = defaultdict(list)
    for split in SPLITS:
        for group_id in groups_by_split[split]:
            group_splits[group_id].append(split)
    shared_groups = {
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    }

    owner_maps = {split: owners_for(split) for split in SPLITS}
    alignment_moves = []
    for group_id in sorted(shared_groups):
        member_splits = [split for split in SPLITS if group_id in groups_by_split[split]]
        canonical_split = member_splits[0]
        target_client = int(owner_maps[canonical_split][group_id])
        for split in member_splits[1:]:
            current_client = int(owner_maps[split][group_id])
            if current_client == target_client:
                continue
            group_ids = set(groups_by_split[split][group_id])
            aligned[split][current_client] = [
                image_id for image_id in aligned[split][current_client]
                if image_id not in group_ids
            ]
            aligned[split][target_client].extend(sorted(group_ids))
            owner_maps[split][group_id] = target_client
            alignment_moves.append({
                "source_group_id": group_id,
                "split": split,
                "from_client": current_client,
                "to_client": target_client,
                "images": len(group_ids),
            })

    balance_moves = []
    for split in SPLITS:
        groups = groups_by_split[split]
        largest_group = max(map(len, groups.values()))
        for _ in range(len(groups) + 1):
            counts = [len(values) for values in aligned[split]]
            current_gap = max(counts) - min(counts)
            if current_gap <= largest_group:
                break
            max_client = min(
                client_id for client_id, count in enumerate(counts)
                if count == max(counts)
            )
            min_client = min(
                client_id for client_id, count in enumerate(counts)
                if count == min(counts)
            )
            current_owners = owners_for(split)
            candidates = []
            for group_id, owner in current_owners.items():
                if owner != max_client or group_id in shared_groups:
                    continue
                size = len(groups[group_id])
                proposed = list(counts)
                proposed[max_client] -= size
                proposed[min_client] += size
                proposed_gap = max(proposed) - min(proposed)
                if proposed_gap < current_gap:
                    candidates.append((proposed_gap, abs(
                        proposed[max_client] - proposed[min_client]
                    ), group_id, size))
            if not candidates:
                raise RuntimeError(
                    f"{split}: cannot restore quantity balance without moving a "
                    "cross-split shared source group"
                )
            _, _, group_id, size = min(candidates)
            group_ids = set(groups[group_id])
            aligned[split][max_client] = [
                image_id for image_id in aligned[split][max_client]
                if image_id not in group_ids
            ]
            aligned[split][min_client].extend(sorted(group_ids))
            balance_moves.append({
                "source_group_id": group_id,
                "split": split,
                "from_client": max_client,
                "to_client": min_client,
                "images": size,
            })
        else:
            raise RuntimeError(f"{split}: cross-split owner balance repair did not converge")
        for client in aligned[split]:
            client.sort()
        validate_source_group_assignments(coco_by_split[split], aligned[split])

    final_owners = {split: owners_for(split) for split in SPLITS}
    conflicts = []
    for group_id in sorted(shared_groups):
        values = {
            final_owners[split][group_id]
            for split in SPLITS if group_id in final_owners[split]
        }
        if len(values) != 1:
            conflicts.append(group_id)
    if conflicts:
        raise RuntimeError(
            f"Cross-split source groups still cross clients: {conflicts[:10]}"
        )

    return aligned, {
        "schema_version": 1,
        "policy": "train_then_val_then_test_global_source_group_client_owner_v1",
        "shared_source_groups": len(shared_groups),
        "alignment_moves": alignment_moves,
        "quantity_balance_moves": balance_moves,
        "cross_split_client_owner_conflicts": 0,
        "per_split": {
            split: {
                "client_image_counts": [len(values) for values in aligned[split]],
                "largest_source_group_images": max(
                    map(len, groups_by_split[split].values())
                ),
            }
            for split in SPLITS
        },
    }


def partition_images_iid(
    coco: dict,
    num_clients: int,
    seed: int,
    *,
    group_atomic: bool = False,
) -> List[List[int]]:
    """Seeded label-blind split without replacement, keeping source groups atomic.

    This is the IID baseline. Unlike the Dirichlet target-deficit allocator, it
    does not use labels when assigning images, so it does not artificially
    stratify away the finite-sample variation expected from random sampling.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if group_atomic:
        groups = _source_groups(coco)
        if len(groups) < num_clients:
            raise ValueError("Every client must receive at least one source group")
        rng = np.random.default_rng(seed)
        tie_noise = {
            group_id: float(rng.random()) for group_id in sorted(groups)
        }
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (
                -len(groups[group_id]), tie_noise[group_id], group_id,
            ),
        )
        assignments = [[] for _ in range(num_clients)]
        assigned_counts = np.zeros(num_clients, dtype=np.int64)
        for group_id in ordered_groups:
            minimum = int(assigned_counts.min())
            candidates = np.flatnonzero(assigned_counts == minimum)
            chosen = int(rng.choice(candidates))
            assignments[chosen].extend(groups[group_id])
            assigned_counts[chosen] += len(groups[group_id])
        for client in assignments:
            rng.shuffle(client)
        validation = validate_source_group_assignments(coco, assignments)
        if (
            validation["max_min_image_gap"]
            > validation["largest_source_group_images"]
        ):
            raise RuntimeError(
                "IID source-group balance exceeds the indivisible-group bound"
            )
        return assignments

    image_ids = np.asarray([int(image["id"]) for image in coco["images"]], dtype=np.int64)
    if len(image_ids) < num_clients:
        raise ValueError("Every client must receive at least one image")
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    assignments = [chunk.astype(int).tolist() for chunk in np.array_split(image_ids, num_clients)]
    flat = [image_id for client in assignments for image_id in client]
    if len(flat) != len(set(flat)) or set(flat) != set(map(int, image_ids)):
        raise RuntimeError("IID client partition is not a disjoint cover")
    return assignments


def partition_images(
    coco: dict,
    num_clients: int,
    proportions: Dict[str, List[float]],
    seed: int,
    *,
    group_atomic: bool = False,
) -> List[List[int]]:
    """Assign whole source groups toward Dirichlet targets with size control.

    Each image contributes its complete multi-class instance histogram to the
    target-deficit score, so an image is never duplicated across clients.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    cat_ids = sorted(category_mapping(coco))
    histograms, totals = _image_histograms(coco, cat_ids)
    image_ids = list(histograms)
    if len(image_ids) < num_clients:
        raise ValueError("Every client must receive at least one image")
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    # Hard capacity makes alpha a label-skew control rather than a hidden
    # quantity-skew control. The realized distributions are stored in metadata.
    base, remainder = divmod(len(image_ids), num_clients)
    capacities = np.array([base + (idx < remainder) for idx in range(num_clients)], dtype=int)

    prop_matrix = np.zeros((num_clients, len(cat_ids) + 1), dtype=np.float64)
    for class_idx, key in enumerate([str(cid) for cid in cat_ids] + [BACKGROUND_KEY]):
        values = np.asarray(proportions[key], dtype=np.float64)
        if (
            len(values) != num_clients
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or values.sum() <= 0
        ):
            raise ValueError(f"Invalid proportions for {key}: {values}")
        prop_matrix[:, class_idx] = values / values.sum()
    target = prop_matrix * totals[None, :]

    if group_atomic:
        groups = _source_groups(coco)
        if len(groups) < num_clients:
            raise ValueError("Every client must receive at least one source group")
        group_histograms = {
            group_id: sum(
                (histograms[image_id] for image_id in group_image_ids),
                start=np.zeros(len(cat_ids) + 1, dtype=np.float64),
            )
            for group_id, group_image_ids in groups.items()
        }
        rarity = 1.0 / np.maximum(totals, 1.0)
        tie_noise = {
            group_id: float(rng.random()) for group_id in sorted(groups)
        }
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (
                -float((group_histograms[group_id] * rarity).sum()),
                -len(groups[group_id]),
                tie_noise[group_id],
                group_id,
            ),
        )
        assigned_groups = [[] for _ in range(num_clients)]
        assigned_hist = np.zeros_like(target)
        assigned_count = np.zeros(num_clients, dtype=np.int64)
        normalizer = np.maximum(totals, 1.0)
        target_image_count = len(image_ids) / num_clients
        largest_group = max(map(len, groups.values()))
        soft_capacity = math.ceil(target_image_count) + largest_group - 1

        for group_index, group_id in enumerate(ordered_groups):
            group_image_ids = groups[group_id]
            hist = group_histograms[group_id]
            empty_clients = np.flatnonzero(assigned_count == 0)
            remaining_groups = len(ordered_groups) - group_index
            if len(empty_clients) and remaining_groups == len(empty_clients):
                candidates = empty_clients
            else:
                fitting = np.flatnonzero(
                    assigned_count + len(group_image_ids) <= soft_capacity
                )
                candidates = fitting if len(fitting) else np.arange(num_clients)
            scores = []
            for client_idx in candidates:
                deficit = target[client_idx] - assigned_hist[client_idx]
                class_score = float((hist * deficit / normalizer).sum())
                capacity_score = float(
                    (target_image_count - assigned_count[client_idx])
                    / max(target_image_count, 1.0)
                )
                scores.append(class_score + 1e-2 * capacity_score)
            best_score = max(scores)
            best_candidates = [
                int(candidates[index]) for index, score in enumerate(scores)
                if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
            ]
            chosen = int(rng.choice(best_candidates))
            assigned_groups[chosen].append(group_id)
            assigned_hist[chosen] += hist
            assigned_count[chosen] += len(group_image_ids)

        # The class-deficit objective alone can leave one client much smaller
        # than the others. Repair whole-group moves until the image-count gap is
        # no larger than the largest indivisible source group. Every move is
        # from a currently largest client to a currently smallest client. With
        # tied extrema, the first move can leave the global gap unchanged, so
        # require a non-increasing gap and a strictly decreasing squared-count
        # potential. This finite integer potential guarantees progress. Prefer
        # the lowest resulting gap and then the best target fit deterministically.
        while int(assigned_count.max() - assigned_count.min()) > largest_group:
            current_gap = int(assigned_count.max() - assigned_count.min())
            current_balance_error = float((
                (assigned_count - target_image_count) ** 2
            ).sum())
            donors = np.flatnonzero(assigned_count == assigned_count.max())
            receivers = np.flatnonzero(assigned_count == assigned_count.min())
            moves = []
            for donor in donors:
                donor = int(donor)
                if len(assigned_groups[donor]) <= 1:
                    continue
                for group_id in assigned_groups[donor]:
                    group_size = len(groups[group_id])
                    hist = group_histograms[group_id]
                    for receiver in receivers:
                        receiver = int(receiver)
                        candidate_counts = assigned_count.copy()
                        candidate_counts[donor] -= group_size
                        candidate_counts[receiver] += group_size
                        candidate_gap = int(
                            candidate_counts.max() - candidate_counts.min()
                        )
                        candidate_balance_error = float((
                            (candidate_counts - target_image_count) ** 2
                        ).sum())
                        if (
                            candidate_gap > current_gap
                            or candidate_balance_error
                            >= current_balance_error - 1e-12
                        ):
                            continue
                        donor_hist = assigned_hist[donor] - hist
                        receiver_hist = assigned_hist[receiver] + hist
                        current_pair_error = float((
                            ((assigned_hist[donor] - target[donor]) / normalizer) ** 2
                        ).sum() + (
                            ((assigned_hist[receiver] - target[receiver]) / normalizer) ** 2
                        ).sum())
                        candidate_pair_error = float((
                            ((donor_hist - target[donor]) / normalizer) ** 2
                        ).sum() + (
                            ((receiver_hist - target[receiver]) / normalizer) ** 2
                        ).sum())
                        class_error_delta = candidate_pair_error - current_pair_error
                        moves.append((
                            candidate_gap,
                            class_error_delta,
                            candidate_balance_error,
                            group_id,
                            donor,
                            receiver,
                        ))
            if not moves:
                raise RuntimeError(
                    "Unable to repair source-group client quantity imbalance"
                )
            _, _, _, group_id, donor, receiver = min(moves)
            assigned_groups[donor].remove(group_id)
            assigned_groups[receiver].append(group_id)
            hist = group_histograms[group_id]
            group_size = len(groups[group_id])
            assigned_hist[donor] -= hist
            assigned_hist[receiver] += hist
            assigned_count[donor] -= group_size
            assigned_count[receiver] += group_size

        assignments = [
            [
                image_id
                for group_id in client_group_ids
                for image_id in groups[group_id]
            ]
            for client_group_ids in assigned_groups
        ]
        for client in assignments:
            rng.shuffle(client)
        validation = validate_source_group_assignments(coco, assignments)
        if (
            validation["max_min_image_gap"]
            > validation["largest_source_group_images"]
        ):
            raise RuntimeError(
                "Source-group quantity balance exceeds the indivisible-group bound"
            )
        return assignments

    # Allocate information-rich/rare-class images first.
    rarity = 1.0 / np.maximum(totals, 1.0)
    tie_noise = {image_id: float(rng.random()) for image_id in image_ids}
    image_ids.sort(
        key=lambda image_id: (
            float((histograms[image_id] * rarity).sum()),
            tie_noise[image_id],
        ),
        reverse=True,
    )

    assignments = [[] for _ in range(num_clients)]
    assigned_hist = np.zeros_like(target)
    assigned_count = np.zeros(num_clients, dtype=int)
    normalizer = np.maximum(totals, 1.0)

    for image_id in image_ids:
        hist = histograms[image_id]
        candidates = np.where(assigned_count < capacities)[0]
        if len(candidates) == 0:
            raise RuntimeError("Partition capacity accounting failed")
        scores = []
        for client_idx in candidates:
            deficit = target[client_idx] - assigned_hist[client_idx]
            class_score = float((hist * deficit / normalizer).sum())
            capacity_score = float(
                (capacities[client_idx] - assigned_count[client_idx]) / max(capacities[client_idx], 1)
            )
            scores.append(class_score + 1e-3 * capacity_score)
        best_score = max(scores)
        best_candidates = [
            int(candidates[idx]) for idx, score in enumerate(scores)
            if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
        ]
        chosen = int(rng.choice(best_candidates))
        assignments[chosen].append(image_id)
        assigned_hist[chosen] += hist
        assigned_count[chosen] += 1

    flat = [image_id for client in assignments for image_id in client]
    if len(flat) != len(set(flat)) or set(flat) != set(image_ids):
        raise RuntimeError("Client partition is not a disjoint cover of the images")
    for client in assignments:
        rng.shuffle(client)
    return assignments


def split_statistics(coco: dict, assignments: List[List[int]]) -> List[dict]:
    cat_ids = sorted(category_mapping(coco))
    histograms, global_totals = _image_histograms(coco, cat_ids)
    global_foreground = global_totals[:-1]
    global_distribution = (
        global_foreground / global_foreground.sum()
        if global_foreground.sum() else global_foreground
    )
    stats = []
    for client_id, image_ids in enumerate(assignments):
        total = np.zeros(len(cat_ids) + 1, dtype=np.float64)
        class_images = np.zeros(len(cat_ids), dtype=np.int64)
        for image_id in image_ids:
            histogram = histograms[image_id]
            total += histogram
            class_images += (histogram[:-1] > 0).astype(np.int64)
        foreground = total[:-1]
        distribution = foreground / foreground.sum() if foreground.sum() else foreground
        nonzero = distribution[distribution > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
        midpoint = 0.5 * (distribution + global_distribution)
        left_mask = distribution > 0
        right_mask = global_distribution > 0
        js_divergence = 0.5 * float(
            (distribution[left_mask] * np.log(distribution[left_mask] / midpoint[left_mask])).sum()
        ) + 0.5 * float(
            (global_distribution[right_mask]
             * np.log(global_distribution[right_mask] / midpoint[right_mask])).sum()
        )
        stats.append({
            "client_id": client_id,
            "num_images": len(image_ids),
            "class_instances": {str(cid): int(foreground[idx]) for idx, cid in enumerate(cat_ids)},
            "class_images": {str(cid): int(class_images[idx]) for idx, cid in enumerate(cat_ids)},
            "background_images": int(total[-1]),
            "class_entropy_nats": entropy,
            "js_divergence_from_global_nats": js_divergence,
            "minimum_class_instances": int(foreground.min()) if len(foreground) else 0,
        })
    return stats


def _clip_bbox_xywh(bbox: Iterable[float], width: int, height: int) -> Optional[Tuple[float, ...]]:
    x, y, box_w, box_h = map(float, bbox)
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + box_w))
    y2 = max(0.0, min(float(height), y + box_h))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    norm_w = (x2 - x1) / width
    norm_h = (y2 - y1) / height
    return cx, cy, norm_w, norm_h


def coco_to_yolo_labels(coco: dict, image_dir: str, output_dir: str,
                        image_ids: Optional[List[int]] = None,
                        cat_id_to_label: Optional[Dict[int, int]] = None) -> str:
    """Create an exact YOLO subset using symlinks and clipped xyxy geometry."""
    image_output = os.path.join(output_dir, "images")
    label_output = os.path.join(output_dir, "labels")
    os.makedirs(image_output, exist_ok=True)
    os.makedirs(label_output, exist_ok=True)

    image_by_id = {int(image["id"]): image for image in coco["images"]}
    anns_by_image = build_image_ann_map(coco)
    if cat_id_to_label is None:
        cat_id_to_label = {cid: idx for idx, cid in enumerate(sorted(category_mapping(coco)))}
    selected = list(image_by_id) if image_ids is None else [int(value) for value in image_ids]

    for image_id in selected:
        if image_id not in image_by_id:
            raise KeyError(f"Unknown image id {image_id}")
        info = image_by_id[image_id]
        filename = str(info["file_name"])
        source = os.path.abspath(os.path.join(image_dir, filename))
        destination = os.path.join(image_output, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.lexists(destination):
            raise FileExistsError(f"Generated destination already exists: {destination}")
        os.symlink(source, destination)

        stem = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_output, stem)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as handle:
            for ann in anns_by_image.get(image_id, []):
                if ann.get("iscrowd", 0):
                    continue
                category_id = int(ann["category_id"])
                if category_id not in cat_id_to_label:
                    raise KeyError(f"Unmapped category id {category_id}")
                clipped = _clip_bbox_xywh(ann["bbox"], int(info["width"]), int(info["height"]))
                if clipped is None:
                    continue
                cx, cy, norm_w, norm_h = clipped
                handle.write(
                    f"{cat_id_to_label[category_id]} {cx:.8f} {cy:.8f} "
                    f"{norm_w:.8f} {norm_h:.8f}\n"
                )
    return image_output


def create_dataset_yaml(output_dir: str, train_img_dir: str, val_img_dir: str,
                        test_img_dir: str, class_names: List[str]) -> str:
    path = os.path.join(output_dir, "dataset.yaml")
    payload = {
        "path": os.path.abspath(output_dir),
        "train": os.path.relpath(train_img_dir, output_dir),
        "val": os.path.relpath(val_img_dir, output_dir),
        "test": os.path.relpath(test_img_dir, output_dir),
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return path


def _float_equal(left, right, tolerance=1e-12):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _validate_generated_yaml_header(path: str, class_names: List[str]):
    """Reject stale YAML metadata even when the generated tree still exists."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Generated dataset YAML is not a mapping: {path}")
    try:
        nc = int(payload["nc"])
        raw_names = payload["names"]
        if isinstance(raw_names, list):
            names = [str(value) for value in raw_names]
        elif isinstance(raw_names, dict):
            indexed = {int(key): str(value) for key, value in raw_names.items()}
            if sorted(indexed) != list(range(len(indexed))):
                raise ValueError("names keys are not contiguous")
            names = [indexed[index] for index in range(len(indexed))]
        else:
            raise TypeError("names must be a list or mapping")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid nc/names in generated dataset YAML: {path}") from error
    if nc != len(class_names) or names != class_names:
        raise ValueError(
            f"Generated YAML class metadata drift in {path}: "
            f"nc/names={nc}/{names!r}, expected={len(class_names)}/{class_names!r}"
        )


def prepare_data(args) -> dict:
    """Load a pre-generated, versioned split and reject configuration drift."""
    if not os.path.isfile(args.split_file):
        raise FileNotFoundError(
            f"Split manifest not found: {args.split_file}\n"
            "Run scripts/prepare_split.py before any training experiment."
        )
    with open(args.split_file, "r", encoding="utf-8") as handle:
        split_info = json.load(handle)
    metadata = split_info.get("metadata", {})
    if int(metadata.get("schema_version", -1)) != 7:
        raise ValueError("Unsupported split schema. Regenerate it with the updated prepare_split.py")
    if metadata.get("crowd_policy") != (
        "require_zero_crowd_annotations_for_YOLO_metric_equivalence"
    ):
        raise ValueError("Split crowd policy is stale; regenerate it with prepare_split.py")
    if metadata.get("category_policy") != AOD4_CATEGORY_POLICY:
        raise ValueError("Split category policy is stale; regenerate it with prepare_split.py")
    stored_category_audit = metadata.get("source_category_audit")
    if not isinstance(stored_category_audit, dict) or set(stored_category_audit) != set(SPLITS):
        raise ValueError(
            "Split source_category_audit must cover train/val/test; regenerate it"
        )
    source_split_policy = metadata.get("source_split_policy")
    if source_split_policy not in SUPPORTED_SOURCE_SPLIT_POLICIES:
        raise ValueError("Split source policy is stale; regenerate it")
    if metadata.get("official_split_preserved") != (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
    ):
        raise ValueError("Split official-preservation marker is inconsistent")
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and (
            not _float_equal(metadata.get("min_bbox_area", float("nan")), 0.0)
            or not _float_equal(metadata.get("min_bbox_side", float("nan")), 0.0)
        )
    ):
        raise ValueError("Official AOD-4 v6 manifests must preserve all annotations")
    if metadata.get("source_identity_policy") != SOURCE_IDENTITY_POLICY:
        raise ValueError("Split source identity policy is stale; regenerate it")
    expected_priority = (
        () if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        else SOURCE_SPLIT_PRIORITY
    )
    if tuple(metadata.get("source_split_priority", ())) != expected_priority:
        raise ValueError("Split source ownership priority is stale; regenerate it")
    if metadata.get("client_partition_unit") != CLIENT_PARTITION_UNIT:
        raise ValueError("Split client partition unit is stale; regenerate it")
    if not metadata.get("image_hash_check_enabled"):
        raise ValueError(
            "Schema-7 AOD-4 splits require per-file SHA-256 verification"
        )
    source_inventory = metadata.get("source_hash_inventory")
    if not isinstance(source_inventory, dict):
        raise ValueError("Split has no source_hash_inventory; regenerate it")
    stored_source_audit = metadata.get("cross_split_source_audit")
    if not isinstance(stored_source_audit, dict):
        raise ValueError("Split has no cross-split source audit; regenerate it")
    stored_post_policy_check = metadata.get(
        "post_policy_cross_split_source_check"
    )
    if not isinstance(stored_post_policy_check, dict):
        raise ValueError("Split has no post-policy source check; regenerate it")
    source_split_counts = metadata.get("source_split_counts")
    if not isinstance(source_split_counts, dict) or set(source_split_counts) != set(SPLITS):
        raise ValueError("Split source_split_counts must cover train/val/test")

    checks = {
        "num_clients": (int(metadata["num_clients"]), int(args.num_clients)),
        "partition_seed": (int(metadata["seed"]), int(args.partition_seed)),
        "partition": (metadata["partition"], args.partition),
        "data_root": (os.path.abspath(metadata["data_root"]), os.path.abspath(args.data_root)),
    }
    mismatches = [f"{key}: split={left!r}, CLI={right!r}" for key, (left, right) in checks.items()
                  if left != right]
    if not _float_equal(metadata["min_bbox_area"], args.min_bbox_area):
        mismatches.append(
            f"min_bbox_area: split={metadata['min_bbox_area']}, CLI={args.min_bbox_area}"
        )
    if not _float_equal(metadata["min_bbox_side"], args.min_bbox_side):
        mismatches.append(
            f"min_bbox_side: split={metadata['min_bbox_side']}, CLI={args.min_bbox_side}"
        )
    if args.partition == "dirichlet" and not _float_equal(
        metadata["dirichlet_alpha"], args.dirichlet_alpha
    ):
        mismatches.append(
            f"dirichlet_alpha: split={metadata['dirichlet_alpha']}, CLI={args.dirichlet_alpha}"
        )
    if mismatches:
        raise ValueError("Split/CLI configuration mismatch:\n  " + "\n  ".join(mismatches))

    expected_partition_algorithm = (
        "random_source_group_lpt_balance_cross_split_owner_v2"
        if args.partition == "iid"
        else "source_group_target_deficit_balance_cross_split_owner_v2"
    )
    if metadata.get("partition_algorithm") != expected_partition_algorithm:
        raise ValueError(
            "Split partition algorithm is stale; regenerate it with prepare_split.py"
        )

    class_names = list(metadata["class_names"])
    if len(class_names) != args.num_classes:
        raise ValueError(
            f"Split contains {len(class_names)} classes but --num_classes={args.num_classes}"
        )

    # Verify source annotation/category state and reconstruct the selected
    # official-split policy. Image IDs remain split-local throughout.
    expected_declared_categories = None
    expected_target_categories = None
    canonical_coco_by_split = {}
    current_official_counts = {}
    for split in SPLITS:
        annotation_path = os.path.join(args.data_root, split, "_annotations.coco.json")
        if annotation_sha256(annotation_path) != metadata["annotation_sha256"][split]:
            raise ValueError(
                f"{split} annotation JSON changed after split generation; regenerate the split"
            )
        source_coco = load_coco_annotations(annotation_path)
        annotated_image_ids = {
            int(annotation["image_id"])
            for annotation in source_coco["annotations"]
        }
        current_official_counts[split] = {
            "images": len(source_coco["images"]),
            "annotations": len(source_coco["annotations"]),
            "background_images": sum(
                int(image["id"]) not in annotated_image_ids
                for image in source_coco["images"]
            ),
        }
        crowd_count = sum(
            int(bool(annotation.get("iscrowd", 0)))
            for annotation in source_coco["annotations"]
        )
        if crowd_count:
            raise ValueError(
                f"{split} contains {crowd_count} crowd annotations; regenerate with a "
                "representation that preserves COCO crowd-ignore semantics"
            )
        current_declared = category_mapping(source_coco)
        if expected_declared_categories is None:
            expected_declared_categories = current_declared
        elif current_declared != expected_declared_categories:
            raise ValueError(f"{split} category mapping differs from train")
        canonical_source, current_category_audit = canonicalize_aod4_categories(source_coco)
        if current_category_audit != stored_category_audit[split]:
            raise ValueError(
                f"{split} source category audit differs from the immutable manifest"
            )
        current_targets = category_mapping(canonical_source)
        if expected_target_categories is None:
            expected_target_categories = current_targets
        elif current_targets != expected_target_categories:
            raise ValueError(f"{split} canonical AOD-4 target mapping differs from train")
        current_official_counts[split]["class_annotations"] = {
            current_targets[category_id]: sum(
                int(annotation["category_id"]) == category_id
                for annotation in canonical_source["annotations"]
            )
            for category_id in sorted(current_targets)
        }
        actual_source_counts = {
            "images": len(canonical_source["images"]),
            "annotations": len(canonical_source["annotations"]),
        }
        stored_source_counts = source_split_counts[split]
        if {
            key: int(stored_source_counts[key]) for key in actual_source_counts
        } != actual_source_counts:
            raise ValueError(
                f"{split} source_split_counts differ from the canonical source COCO"
            )
        canonical_coco_by_split[split] = canonical_source

    official_count_gate = metadata.get("official_count_gate")
    if (
        source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        and not getattr(args, "_test_allow_unpinned_official_counts", False)
        and not (
            isinstance(official_count_gate, dict)
            and official_count_gate.get("enabled") is True
            and official_count_gate.get("passed") is True
            and official_count_gate.get("expected") == AOD4_V6_OFFICIAL_COUNTS
            and official_count_gate.get("actual") == current_official_counts
            and current_official_counts == AOD4_V6_OFFICIAL_COUNTS
        )
    ):
        raise ValueError(
            "Official AOD-4 v6 count gate is absent, disabled, or inconsistent"
        )

    raw_image_count = sum(
        len(canonical_coco_by_split[split]["images"]) for split in SPLITS
    )
    decode_check = metadata.get("image_decode_check")
    if (
        not isinstance(decode_check, dict)
        or decode_check.get("enabled") is not True
        or decode_check.get("method") != "pillow_verify_then_full_pixel_load"
        or int(decode_check.get("images_checked", -1)) != raw_image_count
        or int(decode_check.get("full_pixel_decodes", -1)) != raw_image_count
        or int(decode_check.get("dimension_mismatches", -1)) != 0
    ):
        raise ValueError(
            "Split lacks a complete full-pixel source-image decode audit; regenerate it"
        )

    ordered_category_ids = sorted(expected_target_categories)
    source_class_names = [expected_target_categories[key] for key in ordered_category_ids]
    source_label_map = {str(key): index for index, key in enumerate(ordered_category_ids)}
    stored_label_map = {
        str(key): int(value) for key, value in metadata["cat_id_to_label"].items()
    }
    if source_class_names != class_names or source_label_map != stored_label_map:
        raise ValueError(
            "Manifest class names/category mapping differ from the source COCO annotations"
        )

    # Tree verification binds the manifest's per-file inventory to current raw
    # bytes. The stored individual hashes can then safely reproduce components
    # without hashing the 1.8 GB tree a second time in this run.
    split_manifest_digest = _verify_source_image_trees(args, metadata)
    validate_source_hash_inventory(canonical_coco_by_split, source_inventory)
    source_selected_coco, current_source_audit = apply_source_split_policy(
        canonical_coco_by_split,
        source_inventory,
        policy=source_split_policy,
    )
    if current_source_audit != stored_source_audit:
        raise ValueError(
            "Recomputed cross-split source audit differs from the immutable manifest"
        )
    if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL:
        if int(current_source_audit.get("excluded", {}).get("images", -1)) != 0:
            raise ValueError("Official AOD-4 v6 policy must exclude zero images")
        for split in SPLITS:
            if (
                current_source_audit["before"]["per_split"][split]
                != current_source_audit["after"]["per_split"][split]
            ):
                raise ValueError(
                    f"Official AOD-4 v6 membership changed for {split}"
                )
    expected_duplicate_counts = (
        {
            "filename_cross_split_duplicates": int(
                current_source_audit["raw_collision_signals"]
                ["exact_filename_cross_split_keys"]
            ),
            "roboflow_source_key_cross_split_duplicates": int(
                current_source_audit["raw_collision_signals"]
                ["roboflow_source_key_cross_split_keys"]
            ),
            "sha256_cross_split_duplicates": int(
                current_source_audit["raw_collision_signals"]
                ["exact_sha256_cross_split_hashes"]
            ),
            "source_group_cross_split_duplicates": int(
                current_source_audit["before"]["cross_split_source_groups"]
            ),
        }
        if source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
        else {
            "filename_cross_split_duplicates": 0,
            "roboflow_source_key_cross_split_duplicates": 0,
            "sha256_cross_split_duplicates": 0,
            "source_group_cross_split_duplicates": 0,
        }
    )
    if (
        stored_post_policy_check.get("policy") != source_split_policy
        or stored_post_policy_check.get("identity_policy") != SOURCE_IDENTITY_POLICY
        or any(
            int(stored_post_policy_check.get(key, -1)) != value
            for key, value in expected_duplicate_counts.items()
        )
        or stored_post_policy_check.get("source_hash_inventory_sha256")
        != source_inventory.get("inventory_sha256")
    ):
        raise ValueError("Stored post-policy source check is inconsistent")
    selected_coco_by_split = {
        split: filter_annotations(
            source_selected_coco[split],
            args.min_bbox_area,
            args.min_bbox_side,
            drop_empty_images=False,
        )
        for split in SPLITS
    }

    seen_group_splits = {}
    for split in SPLITS:
        for image in selected_coco_by_split[split]["images"]:
            group_id = image.get(SOURCE_GROUP_FIELD)
            previous = seen_group_splits.setdefault(group_id, split)
            if (
                source_split_policy == SOURCE_SPLIT_POLICY_EXCLUSIVE
                and previous != split
            ):
                raise ValueError(
                    f"Retained source group {group_id} spans {previous}/{split}"
                )
        actual_counts = {
            "images": len(selected_coco_by_split[split]["images"]),
            "annotations": len(selected_coco_by_split[split]["annotations"]),
        }
        stored_counts = metadata["split_counts"][split]
        if {key: int(stored_counts[key]) for key in actual_counts} != actual_counts:
            raise ValueError(
                f"{split} split_counts differ from the selected official-split policy"
            )
        if (
            source_split_policy == SOURCE_SPLIT_POLICY_OFFICIAL
            and actual_counts != {
                "images": int(source_split_counts[split]["images"]),
                "annotations": int(source_split_counts[split]["annotations"]),
            }
        ):
            raise ValueError(
                f"Official AOD-4 v6 counts changed for {split}: {actual_counts}"
            )

    split_stem = os.path.splitext(os.path.basename(args.split_file))[0]
    yolo_dir = os.path.join(os.path.dirname(os.path.abspath(args.split_file)),
                            split_stem.replace("split_", "yolo_", 1))
    clients = sorted(split_info["clients"], key=lambda item: int(item["client_id"]))
    client_ids = [int(client["client_id"]) for client in clients]
    if client_ids != list(range(args.num_clients)):
        raise ValueError(
            f"Manifest client ids must be contiguous 0..{args.num_clients - 1}: {client_ids}"
        )
    realized_metadata = metadata.get("realized_partition_statistics")
    quantity_metadata = metadata.get("quantity_balance")
    if not isinstance(realized_metadata, dict) or set(realized_metadata) != set(SPLITS):
        raise ValueError("Split realized_partition_statistics must cover train/val/test")
    if not isinstance(quantity_metadata, dict) or set(quantity_metadata) != set(SPLITS):
        raise ValueError("Split quantity_balance must cover train/val/test")
    initial_canonical_assignments = {}
    for split in SPLITS:
        split_seed = int(metadata["seed"]) + {
            "train": 0, "val": 1001, "test": 2001,
        }[split]
        if metadata["partition"] == "iid":
            initial_canonical_assignments[split] = partition_images_iid(
                selected_coco_by_split[split],
                args.num_clients,
                split_seed,
                group_atomic=True,
            )
        else:
            initial_canonical_assignments[split] = partition_images(
                selected_coco_by_split[split],
                args.num_clients,
                metadata["client_target_proportions"],
                split_seed,
                group_atomic=True,
            )
    canonical_assignments_by_split, current_cross_client_report = (
        align_cross_split_source_group_owners(
            selected_coco_by_split,
            initial_canonical_assignments,
        )
    )
    if current_cross_client_report != metadata.get(
        "cross_split_client_source_group_check"
    ):
        raise ValueError(
            "Cross-split source-group client ownership differs from the manifest"
        )
    for split in SPLITS:
        per_client_ids = []
        for client in clients:
            block = client.get("splits", {}).get(split)
            if not isinstance(block, dict):
                raise ValueError(f"Client {client['client_id']} has no {split} split block")
            ids = [int(image_id) for image_id in block.get("image_ids", [])]
            if len(ids) != len(set(ids)) or len(ids) != int(block.get("num_images", -1)):
                raise ValueError(
                    f"Client {client['client_id']} {split} image IDs/num_images mismatch"
                )
            per_client_ids.append(ids)
        all_ids = [image_id for ids in per_client_ids for image_id in ids]
        expected_ids = {
            int(image["id"]) for image in selected_coco_by_split[split]["images"]
        }
        if len(all_ids) != len(expected_ids) or set(all_ids) != expected_ids:
            raise ValueError(
                f"Manifest {split} assignments are not a disjoint cover: "
                f"entries={len(all_ids)}, unique={len(set(all_ids))}, "
                f"expected={len(expected_ids)}"
            )
        group_validation = validate_source_group_assignments(
            selected_coco_by_split[split], per_client_ids
        )

        canonical_assignments = canonical_assignments_by_split[split]
        if per_client_ids != canonical_assignments:
            raise ValueError(
                f"{split} manifest assignment differs from the deterministic "
                "source-group-atomic partition"
            )

        recomputed_stats = split_statistics(
            selected_coco_by_split[split], per_client_ids
        )
        if realized_metadata[split] != recomputed_stats:
            raise ValueError(
                f"{split} realized partition statistics differ from assignments"
            )
        for client, row, group_count in zip(
            clients,
            recomputed_stats,
            group_validation["client_source_group_counts"],
        ):
            block = client["splits"][split]
            for key, expected in row.items():
                if key == "client_id":
                    continue
                actual = block.get(key)
                equal = (
                    _float_equal(actual, expected)
                    if isinstance(expected, float)
                    else actual == expected
                )
                if not equal:
                    raise ValueError(
                        f"Client {client['client_id']} {split} statistic "
                        f"{key!r} differs from assignments"
                    )
            if int(block.get("num_source_groups", -1)) != int(group_count):
                raise ValueError(
                    f"Client {client['client_id']} {split} source-group count mismatch"
                )

        sizes = group_validation["client_image_counts"]
        expected_balance = {
            "min_images": min(sizes),
            "max_images": max(sizes),
            "max_min_gap": group_validation["max_min_image_gap"],
            "min_source_groups": min(
                group_validation["client_source_group_counts"]
            ),
            "max_source_groups": max(
                group_validation["client_source_group_counts"]
            ),
            "largest_source_group_images": group_validation[
                "largest_source_group_images"
            ],
            "balance_bound_images": group_validation[
                "largest_source_group_images"
            ],
        }
        stored_balance = quantity_metadata[split]
        if not isinstance(stored_balance, dict) or any(
            int(stored_balance.get(key, -1)) != int(value)
            for key, value in expected_balance.items()
        ):
            raise ValueError(f"{split} quantity-balance metadata is inconsistent")
        if (
            stored_balance.get("source_group_atomic") is not True
            or stored_balance.get("balance_bound_satisfied") is not True
            or group_validation["max_min_image_gap"]
            > group_validation["largest_source_group_images"]
        ):
            raise ValueError(
                f"{split} source-group atomicity/quantity-balance bound failed"
            )

    client_yamls = []
    client_sizes = []
    client_eval_sizes = []
    client_train_ids = []
    for client in clients:
        client_id = int(client["client_id"])
        yaml_path = os.path.join(yolo_dir, f"client_{client_id}", "dataset.yaml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Missing generated client YAML: {yaml_path}")
        _validate_generated_yaml_header(yaml_path, class_names)
        client_yamls.append(yaml_path)
        train_ids = [int(value) for value in client["splits"]["train"]["image_ids"]]
        client_train_ids.append(train_ids)
        client_sizes.append(len(train_ids))
        client_eval_sizes.append({
            "val": int(client["splits"]["val"]["num_images"]),
            "test": int(client["splits"]["test"]["num_images"]),
        })

    full_yaml = os.path.join(yolo_dir, "full", "dataset.yaml")
    if not os.path.isfile(full_yaml):
        raise FileNotFoundError(f"Missing generated pooled YAML: {full_yaml}")
    _validate_generated_yaml_header(full_yaml, class_names)
    expected_yolo_digest = metadata.get("generated_yolo_tree_sha256")
    actual_yolo_digest = generated_yolo_tree_sha256(yolo_dir)
    if actual_yolo_digest != expected_yolo_digest:
        raise ValueError(
            "Generated YOLO YAML/label/symlink tree changed after split generation; "
            "regenerate the split"
        )

    # Detection evaluation keeps the complete official test split. MIA alone
    # excludes test images whose audited source component appears anywhere in
    # the federated/centralized training pool, so a logical source cannot be
    # labeled both member and non-member.
    global_train_groups = {
        str(image[SOURCE_GROUP_FIELD])
        for image in selected_coco_by_split["train"]["images"]
    }
    test_images_by_id = {
        int(image["id"]): image
        for image in selected_coco_by_split["test"]["images"]
    }
    client_mia_excluded_test_files = []
    client_mia_source_audit = []
    for client in clients:
        test_ids = [
            int(value) for value in client["splits"]["test"]["image_ids"]
        ]
        excluded = sorted(
            str(test_images_by_id[image_id]["file_name"]).replace("\\", "/")
            for image_id in test_ids
            if str(test_images_by_id[image_id][SOURCE_GROUP_FIELD])
            in global_train_groups
        )
        client_mia_excluded_test_files.append(excluded)
        client_mia_source_audit.append({
            "policy": "exclude_test_source_components_present_in_any_train_client",
            "raw_test_images": len(test_ids),
            "excluded_source_overlap_images": len(excluded),
            "eligible_nonmember_images": len(test_ids) - len(excluded),
            "member_nonmember_source_group_intersection": 0,
        })
    return {
        "train_coco": selected_coco_by_split["train"],
        "client_splits": client_train_ids,
        "client_yamls": client_yamls,
        "client_sizes": client_sizes,
        "client_eval_sizes": client_eval_sizes,
        "full_yaml": full_yaml,
        "class_names": class_names,
        "num_train_images": int(metadata["split_counts"]["train"]["images"]),
        "split_metadata": metadata,
        "split_manifest_sha256": split_manifest_digest,
        "client_mia_excluded_test_files": client_mia_excluded_test_files,
        "client_mia_source_audit": client_mia_source_audit,
    }
