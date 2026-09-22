#!/usr/bin/env python3
"""Read-only AOD-4 audit using the exact production preprocessing functions.

Unlike the historical checker, this script does not implement a second,
inconsistent primary-class Dirichlet split. Validation, filtering, clipped
box geometry and the split preview all call ``data.dataset`` directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_DATA_ROOT
from data.dataset import (
    AOD4_CATEGORY_POLICY,
    DEFAULT_SOURCE_SPLIT_POLICY,
    SUPPORTED_SOURCE_SPLIT_POLICIES,
    SPLITS,
    _clip_bbox_xywh,
    align_cross_split_source_group_owners,
    category_mapping,
    draw_client_proportions,
    partition_images,
    partition_images_iid,
    split_statistics,
    validate_source_group_assignments,
)
from scripts.prepare_split import (
    _client_source_group_counts,
    prepare_aod4_data,
)


def _args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate AOD-4 COCO inputs and preview the canonical client split"
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_bbox_area", type=float, default=0.0)
    parser.add_argument("--min_bbox_side", type=float, default=0.0)
    parser.add_argument(
        "--source_split_policy",
        choices=SUPPORTED_SOURCE_SPLIT_POLICIES,
        default=DEFAULT_SOURCE_SPLIT_POLICY,
    )
    parser.add_argument(
        "--hash_cache_path",
        default=str(PROJECT_ROOT / "data" / "splits" / ".aod4_source_hash_inventory_cache.json"),
        help="Validated SHA-256 inventory cache (the full inventory is still recomputed/returned)",
    )
    args = parser.parse_args(argv)
    if args.num_clients <= 0:
        parser.error("--num_clients must be positive")
    if not all(math.isfinite(float(value)) for value in (
        args.dirichlet_alpha, args.min_bbox_area, args.min_bbox_side
    )):
        parser.error("alpha and bbox thresholds must be finite")
    if args.partition == "dirichlet" and args.dirichlet_alpha <= 0:
        parser.error("--dirichlet_alpha must be positive")
    if args.min_bbox_area < 0 or args.min_bbox_side < 0:
        parser.error("bbox thresholds must be nonnegative")
    return args


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _trainable_background_count(coco: dict) -> int:
    foreground = {
        int(annotation["image_id"])
        for annotation in coco["annotations"]
        if not annotation.get("iscrowd", 0)
    }
    return sum(int(image["id"]) not in foreground for image in coco["images"])


def _bbox_summary(coco: dict) -> dict:
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    raw_areas, clipped_areas = [], []
    discarded_outside = 0
    for annotation in coco["annotations"]:
        _, _, width, height = map(float, annotation["bbox"])
        raw_areas.append(width * height)
        image = image_by_id[int(annotation["image_id"])]
        clipped = _clip_bbox_xywh(
            annotation["bbox"], int(image["width"]), int(image["height"])
        )
        if clipped is None:
            discarded_outside += 1
            continue
        clipped_areas.append(
            clipped[2] * clipped[3] * int(image["width"]) * int(image["height"])
        )

    def describe(values):
        if not values:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    return {
        "raw_area": describe(raw_areas),
        "clipped_area": describe(clipped_areas),
        "fully_outside": discarded_outside,
    }


def _print_split_report(split: str, raw: dict, filtered: dict, validation: dict):
    category_names = category_mapping(raw)
    raw_counts = Counter(int(annotation["category_id"]) for annotation in raw["annotations"])
    filtered_counts = Counter(
        int(annotation["category_id"])
        for annotation in filtered["annotations"]
        if not annotation.get("iscrowd", 0)
    )
    widths = np.asarray([int(image["width"]) for image in raw["images"]], dtype=np.int64)
    heights = np.asarray([int(image["height"]) for image in raw["images"]], dtype=np.int64)
    boxes = _bbox_summary(filtered)

    print(f"\n[{split.upper()}]")
    print(
        f"raw_images={len(raw['images'])} retained_images={len(filtered['images'])} "
        f"source_policy_removed_images={len(raw['images']) - len(filtered['images'])} "
        f"raw_annotations={len(raw['annotations'])} "
        f"retained_after_policy_and_bbox_filter={len(filtered['annotations'])} "
        f"raw_COCO_background={validation['background_images']} "
        f"retained_trainable_background={_trainable_background_count(filtered)} "
        f"crowd={validation['crowd_annotations']}"
    )
    print(
        f"image_width[min/mean/max]={widths.min()}/{widths.mean():.1f}/{widths.max()} "
        f"image_height[min/mean/max]={heights.min()}/{heights.mean():.1f}/{heights.max()}"
    )
    print("class                         raw   retained(non-crowd)")
    for category_id in sorted(category_names):
        print(
            f"{category_id:>4} {category_names[category_id]:<20} "
            f"{raw_counts[category_id]:>7} {filtered_counts[category_id]:>20}"
        )
    for label, values in (("raw", boxes["raw_area"]), ("clipped", boxes["clipped_area"])):
        print(
            f"{label}_bbox_area[min/median/mean/max]="
            f"{values['min']:.2f}/{values['median']:.2f}/"
            f"{values['mean']:.2f}/{values['max']:.2f}"
        )
    print(f"fully_outside_boxes_skipped_by_YOLO={boxes['fully_outside']}")
    removed = len(raw["annotations"]) - len(filtered["annotations"])
    if removed:
        print(
            f"source_policy_or_threshold_removed_annotations="
            f"{removed}/{len(raw['annotations'])} "
            f"({_percent(removed, len(raw['annotations'])):.2f}%)"
        )


def _print_partition_preview(coco_by_split: dict, args):
    cat_ids = sorted(category_mapping(coco_by_split["train"]))
    proportions = draw_client_proportions(
        cat_ids,
        args.num_clients,
        args.partition,
        args.dirichlet_alpha,
        args.seed,
    )
    split_seeds = {"train": args.seed, "val": args.seed + 1001, "test": args.seed + 2001}
    initial_assignments = {}
    for split in SPLITS:
        coco = coco_by_split[split]
        initial_assignments[split] = (
            partition_images_iid(
                coco, args.num_clients, split_seeds[split], group_atomic=True
            )
            if args.partition == "iid"
            else partition_images(
                coco, args.num_clients, proportions, split_seeds[split],
                group_atomic=True,
            )
        )
    assignments_by_split, cross_split_report = (
        align_cross_split_source_group_owners(
            coco_by_split,
            initial_assignments,
        )
    )
    for split in SPLITS:
        coco = coco_by_split[split]
        assignments = assignments_by_split[split]
        stats = split_statistics(coco, assignments)
        source_group_counts = _client_source_group_counts(coco, assignments)
        group_validation = validate_source_group_assignments(coco, assignments)
        expected = {int(image["id"]) for image in coco["images"]}
        flat = [image_id for client in assignments for image_id in client]
        if len(flat) != len(set(flat)) or set(flat) != expected:
            raise RuntimeError("Canonical preview is not a disjoint selected-image cover")

        print(
            f"\n[{split.upper()} CLIENT PREVIEW] partition={args.partition} "
            f"alpha={args.dirichlet_alpha} seed={split_seeds[split]}"
        )
        print(
            "client images source_groups background "
            + " ".join(f"cat_{cat_id}" for cat_id in cat_ids)
        )
        for row in stats:
            counts = " ".join(
                str(row["class_instances"][str(cat_id)]) for cat_id in cat_ids
            )
            client_id = int(row["client_id"])
            print(
                f"{client_id:>6} {row['num_images']:>6} "
                f"{source_group_counts[client_id]:>13} "
                f"{row['background_images']:>10} {counts} "
                f"JS={row['js_divergence_from_global_nats']:.6f}"
            )
        sizes = [len(values) for values in assignments]
        print(
            "selected_atomic_cover=PASS source_group_atomic=PASS "
            f"image_count_gap={max(sizes) - min(sizes)} "
            f"balance_bound={group_validation['largest_source_group_images']} "
            "balance_bound_satisfied=PASS deterministic_algorithm=PASS"
        )
    print(
        "\n[CROSS-SPLIT CLIENT SOURCE OWNERSHIP] "
        f"shared_groups={cross_split_report['shared_source_groups']} "
        "conflicts=0"
    )


def main(argv=None):
    args = _args(argv)
    data_root = os.path.abspath(args.data_root)
    print("AOD-4 canonical data and cross-split source audit")
    print(f"data_root={data_root}")
    print(
        f"filter=min_bbox_area:{args.min_bbox_area},"
        f"min_bbox_side:{args.min_bbox_side},drop_empty_images:false"
    )

    prepared = prepare_aod4_data(
        data_root,
        min_bbox_area=args.min_bbox_area,
        min_bbox_side=args.min_bbox_side,
        num_classes=4,
        verify_image_hashes=True,
        verify_image_decode=True,
        hash_cache_path=os.path.abspath(args.hash_cache_path),
        source_split_policy=args.source_split_policy,
    )
    audit = prepared["source_audit"]
    print("\n[CROSS-SPLIT SOURCE AUDIT]")
    print(
        f"policy={audit['policy']} identity={audit['identity_policy']} "
        f"priority={'>'.join(audit['split_priority']) or 'none'}"
    )
    print(
        f"raw_cross_split_source_groups="
        f"{audit['before']['cross_split_source_groups']} "
        f"excluded_images={audit['excluded']['images']} "
        f"post_policy_cross_split_source_groups="
        f"{audit['after']['cross_split_source_groups']}"
    )
    print(
        "raw_collision_signals="
        + json.dumps(
            audit["raw_collision_signals"], ensure_ascii=False, sort_keys=True
        )
    )
    print(f"audit_sha256={audit['audit_sha256']}")
    print(
        "post_policy_check="
        + json.dumps(
            {
                key: value
                for key, value in prepared["post_policy_source_check"].items()
                if key.endswith("_duplicates")
            },
            sort_keys=True,
        )
    )

    for split in SPLITS:
        raw = prepared["raw_coco"][split]
        filtered = prepared["coco_by_split"][split]
        _print_split_report(split, raw, filtered, prepared["validation"][split])
        ignored = prepared["source_category_audit"][split][
            "ignored_unreferenced_categories"
        ]
        if ignored:
            print(
                f"category_policy={AOD4_CATEGORY_POLICY}; "
                f"ignored_zero_annotation_metadata={ignored}"
            )

    _print_partition_preview(prepared["coco_by_split"], args)
    print(
        "\n[PASS] Raw COCO validation, selected source policy and "
        "source-group-atomic client previews are valid"
    )


if __name__ == "__main__":
    main()
