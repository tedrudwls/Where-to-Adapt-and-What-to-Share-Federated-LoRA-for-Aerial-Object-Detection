"""Regression tests for COCO validation, partitioning and YOLO conversion."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from data.dataset import (
    AOD4_CATEGORY_POLICY,
    BACKGROUND_KEY,
    SOURCE_SPLIT_POLICY_OFFICIAL,
    SOURCE_GROUP_FIELD,
    _clip_bbox_xywh,
    align_cross_split_source_group_owners,
    apply_source_leakage_policy,
    apply_source_split_policy,
    build_image_ann_map,
    build_source_hash_inventory,
    canonicalize_aod4_categories,
    category_mapping,
    coco_to_yolo_labels,
    draw_client_proportions,
    filter_annotations,
    generated_yolo_tree_sha256,
    image_tree_sha256,
    image_tree_stat_sha256,
    partition_images,
    partition_images_iid,
    prepare_data,
    source_image_key,
    split_statistics,
    validate_source_group_assignments,
    validate_coco,
)
from scripts.prepare_split import (
    _verify_image_decoding,
    main as prepare_split_main,
)


def _toy_coco(num_images: int = 12) -> dict:
    categories = [
        {"id": 11, "name": "airplane"},
        {"id": 3, "name": "bird"},
        {"id": 27, "name": "drone"},
        {"id": 8, "name": "helicopter"},
    ]
    images = [
        {"id": index + 100, "file_name": f"nested/image_{index:02d}.jpg", "width": 100, "height": 50}
        for index in range(num_images)
    ]
    annotations = []
    annotation_id = 1
    category_ids = [11, 3, 27, 8]
    for index, image in enumerate(images):
        # Every fifth image is an intentional background image. Other images
        # include one or two classes so partitioning is genuinely multi-object.
        if index % 5 == 4:
            continue
        annotations.append({
            "id": annotation_id,
            "image_id": image["id"],
            "category_id": category_ids[index % len(category_ids)],
            "bbox": [5.0, 6.0, 12.0, 10.0],
            "iscrowd": 0,
        })
        annotation_id += 1
        if index % 3 == 0:
            annotations.append({
                "id": annotation_id,
                "image_id": image["id"],
                "category_id": category_ids[(index + 1) % len(category_ids)],
                "bbox": [25.0, 8.0, 8.0, 9.0],
                "iscrowd": 0,
            })
            annotation_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


class CocoValidationTests(unittest.TestCase):
    def test_unused_roboflow_aggregate_category_is_audited_and_projected(self):
        coco = _toy_coco()
        coco["categories"].insert(
            0,
            {"id": 0, "name": "airplane-helicopter-drone-bird"},
        )
        original_annotations = list(coco["annotations"])

        canonical, audit = canonicalize_aod4_categories(coco)

        self.assertEqual(set(category_mapping(canonical)), {3, 8, 11, 27})
        self.assertEqual(audit["ignored_unreferenced_categories"], {
            "0": "airplane-helicopter-drone-bird",
        })
        self.assertEqual(audit["raw_annotation_counts"]["0"], 0)
        self.assertEqual(canonical["annotations"], original_annotations)
        self.assertEqual(len(coco["categories"]), 5, "input COCO must not be mutated")

        proportions = draw_client_proportions(
            sorted(category_mapping(canonical)), 3, "dirichlet", alpha=0.4, seed=42
        )
        self.assertNotIn("0", proportions)
        self.assertEqual(
            set(proportions), {"3", "8", "11", "27", BACKGROUND_KEY}
        )
        self.assertIn("unreferenced", AOD4_CATEGORY_POLICY)

    def test_referenced_non_target_category_fails_instead_of_dropping_gt(self):
        coco = _toy_coco()
        coco["categories"].insert(
            0,
            {"id": 0, "name": "airplane-helicopter-drone-bird"},
        )
        coco["annotations"][0]["category_id"] = 0

        with self.assertRaisesRegex(
            ValueError, "Non-target COCO categories contain ground-truth annotations"
        ):
            canonicalize_aod4_categories(coco)

    def test_missing_or_duplicate_aod4_target_category_fails(self):
        missing = _toy_coco()
        missing["categories"] = [
            category for category in missing["categories"]
            if category["name"] != "drone"
        ]
        missing["annotations"] = [
            annotation for annotation in missing["annotations"]
            if annotation["category_id"] != 27
        ]
        with self.assertRaisesRegex(ValueError, "missing=.*drone"):
            canonicalize_aod4_categories(missing)

        duplicate = _toy_coco()
        duplicate["categories"].append({"id": 99, "name": "Drone"})
        with self.assertRaisesRegex(ValueError, "duplicate=.*drone"):
            canonicalize_aod4_categories(duplicate)

    def test_generated_yolo_tree_digest_covers_labels_yaml_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as source:
            source_a = os.path.join(source, "a.jpg")
            source_b = os.path.join(source, "b.jpg")
            with open(source_a, "wb") as handle:
                handle.write(b"a")
            with open(source_b, "wb") as handle:
                handle.write(b"b")

            label = os.path.join(directory, "client_0", "train", "labels", "a.txt")
            image = os.path.join(directory, "client_0", "train", "images", "a.jpg")
            dataset_yaml = os.path.join(directory, "client_0", "dataset.yaml")
            os.makedirs(os.path.dirname(label), exist_ok=True)
            os.makedirs(os.path.dirname(image), exist_ok=True)
            with open(label, "w", encoding="utf-8") as handle:
                handle.write("0 0.5 0.5 0.1 0.1\n")
            with open(dataset_yaml, "w", encoding="utf-8") as handle:
                handle.write("nc: 1\n")
            os.symlink(source_a, image)

            first = generated_yolo_tree_sha256(directory)
            with open(label, "a", encoding="utf-8") as handle:
                handle.write("0 0.4 0.4 0.1 0.1\n")
            second = generated_yolo_tree_sha256(directory)
            self.assertNotEqual(first, second)

            os.unlink(image)
            os.symlink(source_b, image)
            third = generated_yolo_tree_sha256(directory)
            self.assertNotEqual(second, third)

            cache_path = os.path.join(directory, "labels.cache")
            with open(cache_path, "wb") as handle:
                handle.write(b"runtime-cache")
            self.assertEqual(third, generated_yolo_tree_sha256(directory))

    def test_image_tree_content_and_stat_digests_change_with_source_file(self):
        coco = _toy_coco(2)
        with tempfile.TemporaryDirectory() as directory:
            for image in coco["images"]:
                path = os.path.join(directory, image["file_name"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(b"first")
            content_before = image_tree_sha256(coco, directory)
            stat_before = image_tree_stat_sha256(coco, directory)
            changed = os.path.join(directory, coco["images"][0]["file_name"])
            with open(changed, "wb") as handle:
                handle.write(b"second-longer")
            self.assertNotEqual(content_before, image_tree_sha256(coco, directory))
            self.assertNotEqual(stat_before, image_tree_stat_sha256(coco, directory))

    def test_roboflow_source_key_removes_only_export_hash(self):
        first = "nested/frame_001_jpg.rf.0123456789abcdef0123456789abcdef.jpg"
        second = "nested/frame_001_jpg.rf.ffffffffffffffffffffffffffffffff.jpg"
        self.assertEqual(source_image_key(first), "nested/frame_001_jpg.jpg")
        self.assertEqual(source_image_key(first), source_image_key(second))

    def test_category_mapping_supports_noncontiguous_ids_and_is_strict(self):
        coco = _toy_coco()
        self.assertEqual(
            category_mapping(coco),
            {11: "airplane", 3: "bird", 27: "drone", 8: "helicopter"},
        )

        duplicate = _toy_coco()
        duplicate["categories"].append({"id": 11, "name": "duplicate"})
        with self.assertRaisesRegex(ValueError, "Duplicate category id"):
            category_mapping(duplicate)

        empty_name = _toy_coco()
        empty_name["categories"][0]["name"] = "  "
        with self.assertRaisesRegex(ValueError, "Empty category name"):
            category_mapping(empty_name)

    def test_validate_coco_retains_and_counts_background_images(self):
        coco = _toy_coco(6)
        with tempfile.TemporaryDirectory() as directory:
            for image in coco["images"]:
                path = os.path.join(directory, image["file_name"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(b"not-decoded-by-static-validator")

            stats = validate_coco(coco, directory, "train")
            annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
            self.assertEqual(stats["images"], len(coco["images"]))
            self.assertEqual(
                stats["background_images"],
                len(coco["images"]) - len(annotated_ids),
            )

            mismatched = dict(category_mapping(coco))
            mismatched[11] = "fixed-wing"
            with self.assertRaisesRegex(ValueError, "differs from train"):
                validate_coco(coco, directory, "val", expected_categories=mismatched)

    def test_validate_coco_fails_on_missing_file_unknown_category_and_invalid_box(self):
        coco = _toy_coco(2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_coco(coco, directory, "train")

            for image in coco["images"]:
                path = os.path.join(directory, image["file_name"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(b"x")

            unknown = _toy_coco(2)
            unknown["annotations"][0]["category_id"] = 999
            with self.assertRaisesRegex(ValueError, "unknown category"):
                validate_coco(unknown, directory, "train")

            invalid = _toy_coco(2)
            invalid["annotations"][0]["bbox"] = [0, 0, float("nan"), 2]
            with self.assertRaisesRegex(ValueError, "invalid bounding boxes"):
                validate_coco(invalid, directory, "train")

            outside = _toy_coco(2)
            outside["annotations"][0]["bbox"] = [120, 5, 10, 10]
            with self.assertRaisesRegex(ValueError, "invalid bounding boxes"):
                validate_coco(outside, directory, "train")


class SourceLeakageControlTests(unittest.TestCase):
    def test_official_policy_preserves_all_cross_split_source_members(self):
        categories = _toy_coco(1)["categories"]
        with tempfile.TemporaryDirectory() as data_root:
            coco_by_split = {}
            for split_index, split in enumerate(("train", "val", "test")):
                split_dir = os.path.join(data_root, split)
                os.makedirs(split_dir)
                shared_name = (
                    "shared_jpg.rf."
                    + chr(ord("a") + split_index) * 32
                    + ".jpg"
                )
                unique_name = f"{split}_unique.jpg"
                images = [
                    {"id": 1, "file_name": shared_name, "width": 10, "height": 10},
                    {"id": 2, "file_name": unique_name, "width": 10, "height": 10},
                ]
                for filename in (shared_name, unique_name):
                    with open(os.path.join(split_dir, filename), "wb") as handle:
                        handle.write(f"{split}:{filename}".encode())
                coco_by_split[split] = {
                    "images": images,
                    "annotations": [],
                    "categories": categories,
                }

            inventory = build_source_hash_inventory(coco_by_split, data_root)
            selected, audit = apply_source_split_policy(
                coco_by_split,
                inventory,
                policy=SOURCE_SPLIT_POLICY_OFFICIAL,
            )

            for split in ("train", "val", "test"):
                self.assertEqual(
                    {int(image["id"]) for image in selected[split]["images"]},
                    {1, 2},
                )
                self.assertTrue(all(
                    SOURCE_GROUP_FIELD in image
                    for image in selected[split]["images"]
                ))
            self.assertEqual(audit["before"]["cross_split_source_groups"], 1)
            self.assertEqual(audit["after"]["cross_split_source_groups"], 1)
            self.assertEqual(audit["excluded"]["images"], 0)
            self.assertEqual(audit["split_priority"], [])

    def test_connected_source_and_sha_component_uses_test_priority(self):
        categories = _toy_coco(1)["categories"]
        with tempfile.TemporaryDirectory() as data_root:
            coco_by_split = {}
            file_specs = {
                "train": [
                    (1, "shared_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg", b"train-source"),
                    (2, "train_unique.jpg", b"train-unique"),
                ],
                "val": [
                    (101, "shared_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg", b"bridge"),
                    (102, "val_unique.jpg", b"val-unique"),
                ],
                "test": [
                    # Different basename but exact bytes connect this record to
                    # the val member, making the entire transitive component test-owned.
                    (201, "renamed_bridge.jpg", b"bridge"),
                    (202, "test_unique.jpg", b"test-unique"),
                ],
            }
            for split, specs in file_specs.items():
                split_dir = os.path.join(data_root, split)
                os.makedirs(split_dir)
                images = []
                annotations = []
                for annotation_id, (image_id, filename, content) in enumerate(specs, 1):
                    with open(os.path.join(split_dir, filename), "wb") as handle:
                        handle.write(content)
                    images.append({
                        "id": image_id,
                        "file_name": filename,
                        "width": 10,
                        "height": 10,
                    })
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 11,
                        "bbox": [1, 1, 2, 2],
                        "iscrowd": 0,
                    })
                coco_by_split[split] = {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                }

            original_ids = {
                split: [image["id"] for image in coco["images"]]
                for split, coco in coco_by_split.items()
            }
            inventory = build_source_hash_inventory(coco_by_split, data_root)
            cleaned, audit = apply_source_leakage_policy(coco_by_split, inventory)

            self.assertEqual(
                {image["id"] for image in cleaned["train"]["images"]}, {2}
            )
            self.assertEqual(
                {image["id"] for image in cleaned["val"]["images"]}, {102}
            )
            self.assertEqual(
                {image["id"] for image in cleaned["test"]["images"]}, {201, 202}
            )
            self.assertEqual(audit["before"]["cross_split_source_groups"], 1)
            self.assertEqual(audit["after"]["cross_split_source_groups"], 0)
            self.assertEqual(audit["excluded"]["images"], 2)
            self.assertEqual(
                audit["raw_collision_signals"]["roboflow_source_key_cross_split_keys"],
                1,
            )
            self.assertEqual(
                audit["raw_collision_signals"]["exact_sha256_cross_split_hashes"],
                1,
            )
            for split in ("train", "val", "test"):
                self.assertEqual(
                    [image["id"] for image in coco_by_split[split]["images"]],
                    original_ids[split],
                    "source cleaning must not mutate raw COCO objects",
                )
                self.assertTrue(all(
                    SOURCE_GROUP_FIELD in image for image in cleaned[split]["images"]
                ))

    def test_group_atomic_iid_and_dirichlet_are_deterministic(self):
        coco = _toy_coco(12)
        for index, image in enumerate(coco["images"]):
            image[SOURCE_GROUP_FIELD] = f"group_{index // 2}"

        iid_first = partition_images_iid(coco, 3, 42, group_atomic=True)
        iid_second = partition_images_iid(coco, 3, 42, group_atomic=True)
        self.assertEqual(iid_first, iid_second)
        iid_report = validate_source_group_assignments(coco, iid_first)
        self.assertEqual(iid_report["client_source_group_counts"], [2, 2, 2])

        # IID grouping must remain label-blind.
        relabelled = json.loads(json.dumps(coco))
        for annotation in relabelled["annotations"]:
            annotation["category_id"] = 11
        self.assertEqual(
            iid_first,
            partition_images_iid(relabelled, 3, 42, group_atomic=True),
        )

        cat_ids = sorted(category_mapping(coco))
        proportions = draw_client_proportions(
            cat_ids, 3, "dirichlet", alpha=0.4, seed=42
        )
        dirichlet_first = partition_images(
            coco, 3, proportions, 42, group_atomic=True
        )
        dirichlet_second = partition_images(
            coco, 3, proportions, 42, group_atomic=True
        )
        self.assertEqual(dirichlet_first, dirichlet_second)
        validate_source_group_assignments(coco, dirichlet_first)

    def test_cross_split_shared_source_group_gets_one_client_owner(self):
        coco_by_split = {}
        initial = {}
        for split_index, split in enumerate(("train", "val", "test")):
            coco = _toy_coco(6)
            for image_index, image in enumerate(coco["images"]):
                image["id"] += split_index * 1000
                image[SOURCE_GROUP_FIELD] = (
                    "shared" if image_index == 0 else f"{split}_g{image_index}"
                )
            for annotation in coco["annotations"]:
                annotation["image_id"] += split_index * 1000
            coco_by_split[split] = coco
            initial[split] = partition_images_iid(
                coco, 2, seed=42 + split_index, group_atomic=True
            )

        aligned, report = align_cross_split_source_group_owners(
            coco_by_split, initial
        )

        owners = []
        for split in ("train", "val", "test"):
            shared_image_id = next(
                int(image["id"])
                for image in coco_by_split[split]["images"]
                if image[SOURCE_GROUP_FIELD] == "shared"
            )
            owners.append(next(
                client_id for client_id, image_ids in enumerate(aligned[split])
                if shared_image_id in image_ids
            ))
            validate_source_group_assignments(coco_by_split[split], aligned[split])
        self.assertEqual(len(set(owners)), 1)
        self.assertEqual(report["shared_source_groups"], 1)
        self.assertEqual(report["cross_split_client_owner_conflicts"], 0)

    def test_group_atomic_dirichlet_repairs_quantity_gap_to_group_bound(self):
        categories = _toy_coco(1)["categories"]
        images = []
        annotations = []
        image_id = 1
        for group_index, group_size in enumerate([30] * 16 + [7]):
            for _ in range(group_size):
                images.append({
                    "id": image_id,
                    "file_name": f"image_{image_id}.jpg",
                    "width": 32,
                    "height": 32,
                    SOURCE_GROUP_FIELD: f"group_{group_index:02d}",
                })
                annotations.append({
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 11,
                    "bbox": [1.0, 1.0, 4.0, 4.0],
                    "iscrowd": 0,
                })
                image_id += 1
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        proportions = {
            "3": [1 / 3, 1 / 3, 1 / 3],
            "8": [1 / 3, 1 / 3, 1 / 3],
            "11": [0.0, 0.5, 0.5],
            "27": [1 / 3, 1 / 3, 1 / 3],
            BACKGROUND_KEY: [1 / 3, 1 / 3, 1 / 3],
        }

        assignments = partition_images(
            coco, 3, proportions, seed=42, group_atomic=True
        )
        report = validate_source_group_assignments(coco, assignments)
        self.assertLessEqual(
            report["max_min_image_gap"],
            report["largest_source_group_images"],
        )
        self.assertEqual(
            assignments,
            partition_images(coco, 3, proportions, seed=42, group_atomic=True),
        )

    def test_group_balance_repair_handles_tied_extrema_with_four_clients(self):
        sizes = [
            4, 17, 39, 43, 21, 34, 33, 13, 4, 24, 13, 1, 34, 44, 37,
            15, 31, 46, 38, 10, 3, 9, 9, 2, 41, 18, 37, 34, 7, 33,
            30, 43, 27, 8,
        ]
        categories = [
            {"id": 1, "name": "airplane"},
            {"id": 2, "name": "bird"},
            {"id": 3, "name": "drone"},
            {"id": 4, "name": "helicopter"},
        ]
        images, annotations = [], []
        image_id = annotation_id = 1
        for group_index, group_size in enumerate(sizes):
            for within_group in range(group_size):
                images.append({
                    "id": image_id,
                    "file_name": f"{image_id}.jpg",
                    "width": 10,
                    "height": 10,
                    SOURCE_GROUP_FIELD: f"g{group_index}",
                })
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1 + group_index % 4,
                    "bbox": [1, 1, 2, 2],
                    "iscrowd": 0,
                })
                annotation_id += 1
                if (group_index + within_group) % 7 == 0:
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1 + (group_index + 1) % 4,
                        "bbox": [2, 2, 2, 2],
                        "iscrowd": 0,
                    })
                    annotation_id += 1
                image_id += 1
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        proportions = draw_client_proportions(
            [1, 2, 3, 4], 4, "dirichlet", alpha=0.01, seed=20
        )

        assignments = partition_images(
            coco, 4, proportions, seed=20, group_atomic=True
        )
        report = validate_source_group_assignments(coco, assignments)
        self.assertLessEqual(
            report["max_min_image_gap"],
            report["largest_source_group_images"],
        )
        self.assertEqual(
            assignments,
            partition_images(coco, 4, proportions, seed=20, group_atomic=True),
        )

    def test_full_pixel_decode_rejects_truncated_jpeg(self):
        with tempfile.TemporaryDirectory() as data_root:
            split_dir = os.path.join(data_root, "train")
            os.makedirs(split_dir)
            path = os.path.join(split_dir, "truncated.jpg")
            Image.new("RGB", (128, 128), (12, 34, 56)).save(path, format="JPEG")
            with open(path, "rb") as handle:
                payload = handle.read()
            with open(path, "wb") as handle:
                handle.write(payload[:-20])
            # This fixture deliberately passes Pillow's structural check but
            # fails only when the pixel body is actually decoded.
            with Image.open(path) as decoded:
                decoded.verify()
            with self.assertRaises(OSError):
                with Image.open(path) as decoded:
                    decoded.load()
            coco = {
                "images": [{
                    "id": 1,
                    "file_name": "truncated.jpg",
                    "width": 128,
                    "height": 128,
                }],
                "annotations": [],
                "categories": _toy_coco(1)["categories"],
            }
            with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                _verify_image_decoding({"train": coco}, data_root, True)


class PrepareSplitCategoryIntegrationTests(unittest.TestCase):
    def test_schema7_prepare_and_runtime_preserve_official_sources_and_four_targets(self):
        categories = [
            {"id": 0, "name": "airplane-helicopter-drone-bird"},
            {"id": 1, "name": "airplane"},
            {"id": 2, "name": "bird"},
            {"id": 3, "name": "drone"},
            {"id": 4, "name": "helicopter"},
        ]
        with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as output:
            legacy_manifest = os.path.join(
                output, "split_dirichlet_a0.4_c2_s42.json"
            )
            legacy_yolo_yaml = os.path.join(
                output, "yolo_dirichlet_a0.4_c2_s42", "client_0", "dataset.yaml"
            )
            os.makedirs(os.path.dirname(legacy_yolo_yaml))
            with open(legacy_manifest, "wb") as handle:
                handle.write(b"legacy-manifest-sentinel")
            with open(legacy_yolo_yaml, "wb") as handle:
                handle.write(b"legacy-yolo-sentinel")
            for split_index, split in enumerate(("train", "val", "test")):
                split_dir = os.path.join(data_root, split)
                os.makedirs(split_dir)
                images = []
                annotations = []
                for image_index in range(6):
                    image_id = split_index * 100 + image_index + 1
                    if split == "train" and image_index == 0:
                        filename = (
                            "shared_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
                        )
                    elif split == "val" and image_index == 0:
                        filename = (
                            "shared_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
                        )
                    else:
                        filename = f"{split}_image_{image_index}.jpg"
                    images.append({
                        "id": image_id,
                        "file_name": filename,
                        "width": 64,
                        "height": 48,
                    })
                    if image_index == 1 and split in ("train", "test"):
                        # Exact pixels/encoding under different filenames test
                        # SHA-based component ownership in the full pipeline.
                        color = (250, 1, 1)
                    else:
                        color = (
                            10 + split_index * 70,
                            5 + image_index * 30,
                            20 + split_index * 20 + image_index,
                        )
                    Image.new("RGB", (64, 48), color).save(
                        os.path.join(split_dir, filename), format="JPEG"
                    )
                    annotations.append({
                        "id": split_index * 100 + image_index + 1,
                        "image_id": image_id,
                        "category_id": (image_index % 4) + 1,
                        "bbox": [2.0, 3.0, 10.0, 8.0],
                        "iscrowd": 0,
                    })
                with open(
                    os.path.join(split_dir, "_annotations.coco.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump({
                        "images": images,
                        "annotations": annotations,
                        "categories": categories,
                    }, handle)

            prepare_split_main([
                "--data_root", data_root,
                "--output_dir", output,
                "--partition", "dirichlet",
                "--alpha", "0.4",
                "--num_clients", "2",
                "--seed", "42",
                "--no-enforce_official_counts",
            ])
            with open(legacy_manifest, "rb") as handle:
                self.assertEqual(handle.read(), b"legacy-manifest-sentinel")
            with open(legacy_yolo_yaml, "rb") as handle:
                self.assertEqual(handle.read(), b"legacy-yolo-sentinel")
            split_file = os.path.join(
                output, "split_official_v6_dirichlet_a0.4_c2_s42.json"
            )
            with open(split_file, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            metadata = manifest["metadata"]
            self.assertEqual(metadata["schema_version"], 7)
            self.assertEqual(
                metadata["source_split_policy"], SOURCE_SPLIT_POLICY_OFFICIAL
            )
            self.assertEqual(
                metadata["image_decode_check"]["full_pixel_decodes"], 18
            )
            self.assertEqual(metadata["class_names"], [
                "airplane", "bird", "drone", "helicopter",
            ])
            self.assertEqual(metadata["cat_id_to_label"], {
                "1": 0, "2": 1, "3": 2, "4": 3,
            })
            self.assertNotIn("0", metadata["client_target_proportions"])
            self.assertEqual(metadata["client_partition_unit"], "source_group")
            self.assertEqual(metadata["source_split_counts"]["train"]["images"], 6)
            self.assertEqual(metadata["split_counts"]["train"]["images"], 6)
            self.assertEqual(metadata["split_counts"]["val"]["images"], 6)
            self.assertEqual(metadata["split_counts"]["test"]["images"], 6)
            source_audit = metadata["cross_split_source_audit"]
            self.assertEqual(source_audit["before"]["cross_split_source_groups"], 2)
            self.assertEqual(source_audit["after"]["cross_split_source_groups"], 2)
            self.assertEqual(source_audit["excluded"]["images"], 0)
            self.assertEqual(
                metadata["cross_split_client_source_group_check"]
                ["cross_split_client_owner_conflicts"],
                0,
            )
            post_policy = metadata["post_policy_cross_split_source_check"]
            self.assertEqual(post_policy["source_group_cross_split_duplicates"], 2)
            for split in ("train", "val", "test"):
                self.assertEqual(
                    metadata["source_category_audit"][split]
                    ["ignored_unreferenced_categories"],
                    {"0": "airplane-helicopter-drone-bird"},
                )
                for row in metadata["realized_partition_statistics"][split]:
                    self.assertNotIn("0", row["class_instances"])
                for client in manifest["clients"]:
                    self.assertGreater(
                        int(client["splits"][split]["num_source_groups"]), 0
                    )
                self.assertTrue(
                    metadata["quantity_balance"][split][
                        "balance_bound_satisfied"
                    ]
                )

            args = SimpleNamespace(
                split_file=split_file,
                num_clients=2,
                partition_seed=42,
                partition="dirichlet",
                data_root=data_root,
                min_bbox_area=0.0,
                min_bbox_side=0.0,
                dirichlet_alpha=0.4,
                num_classes=4,
                rehash_source_images=False,
                _test_allow_unpinned_official_counts=True,
            )
            data_info = prepare_data(args)
            self.assertEqual(data_info["class_names"], metadata["class_names"])
            self.assertEqual(
                set(category_mapping(data_info["train_coco"])), {1, 2, 3, 4}
            )
            self.assertEqual(
                sum(
                    row["excluded_source_overlap_images"]
                    for row in data_info["client_mia_source_audit"]
                ),
                1,
            )
            self.assertTrue(all(
                row["member_nonmember_source_group_intersection"] == 0
                for row in data_info["client_mia_source_audit"]
            ))


class FilteringAndGeometryTests(unittest.TestCase):
    def test_annotation_filter_keeps_original_and_newly_empty_images_by_default(self):
        coco = _toy_coco(6)
        original_image_ids = [image["id"] for image in coco["images"]]
        filtered = filter_annotations(
            coco,
            min_bbox_area=10_000.0,
            min_bbox_side=100.0,
            drop_empty_images=False,
        )
        self.assertEqual(filtered["annotations"], [])
        self.assertEqual([image["id"] for image in filtered["images"]], original_image_ids)
        # The input object must not be mutated.
        self.assertGreater(len(coco["annotations"]), 0)

    def test_bbox_is_clipped_as_xyxy_before_normalization(self):
        clipped = _clip_bbox_xywh([-10.0, -5.0, 30.0, 20.0], width=100, height=50)
        self.assertIsNotNone(clipped)
        np.testing.assert_allclose(clipped, (0.1, 0.15, 0.2, 0.3), rtol=0, atol=1e-12)

        right_bottom = _clip_bbox_xywh([90.0, 40.0, 30.0, 20.0], width=100, height=50)
        np.testing.assert_allclose(right_bottom, (0.95, 0.9, 0.1, 0.2), rtol=0, atol=1e-12)
        self.assertIsNone(_clip_bbox_xywh([120.0, 5.0, 10.0, 10.0], 100, 50))
        self.assertIsNone(_clip_bbox_xywh([-20.0, 5.0, 10.0, 10.0], 100, 50))

    def test_yolo_conversion_clips_boxes_skips_crowd_and_writes_empty_background_label(self):
        coco = {
            "categories": [
                {"id": 9, "name": "drone"},
                {"id": 2, "name": "bird"},
            ],
            "images": [
                {"id": 1, "file_name": "nested/foreground.jpg", "width": 100, "height": 50},
                {"id": 2, "file_name": "background.jpg", "width": 100, "height": 50},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 9,
                 "bbox": [-10, -5, 30, 20], "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2,
                 "bbox": [20, 10, 5, 5], "iscrowd": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            for image in coco["images"]:
                path = os.path.join(source, image["file_name"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(b"image-placeholder")

            image_dir = coco_to_yolo_labels(
                coco,
                source,
                output,
                image_ids=[1, 2],
                cat_id_to_label={2: 0, 9: 1},
            )
            self.assertTrue(os.path.islink(os.path.join(image_dir, "nested/foreground.jpg")))
            foreground_label = os.path.join(output, "labels", "nested", "foreground.txt")
            background_label = os.path.join(output, "labels", "background.txt")
            with open(foreground_label, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            self.assertEqual(lines, ["1 0.10000000 0.15000000 0.20000000 0.30000000"])
            with open(background_label, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                coco_to_yolo_labels(
                    coco, source, output, image_ids=[1], cat_id_to_label={2: 0, 9: 1}
                )


class PartitionTests(unittest.TestCase):
    def test_iid_proportions_include_background_and_are_uniform(self):
        proportions = draw_client_proportions([3, 8, 11, 27], 3, "iid", alpha=0.4, seed=42)
        self.assertEqual(set(proportions), {"3", "8", "11", "27", BACKGROUND_KEY})
        for values in proportions.values():
            np.testing.assert_allclose(values, [1 / 3, 1 / 3, 1 / 3])

    def test_iid_assignment_is_seeded_random_balanced_and_label_blind(self):
        coco = _toy_coco(17)
        first = partition_images_iid(coco, 3, seed=42)
        second = partition_images_iid(coco, 3, seed=42)
        different = partition_images_iid(coco, 3, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        flat = [image_id for client in first for image_id in client]
        self.assertEqual(set(flat), {int(image["id"]) for image in coco["images"]})
        self.assertEqual(len(flat), len(set(flat)))
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)

    def test_partition_is_deterministic_balanced_disjoint_atomic_cover(self):
        coco = _toy_coco(17)
        cat_ids = sorted(category_mapping(coco))
        proportions = draw_client_proportions(cat_ids, 3, "dirichlet", alpha=0.4, seed=123)

        first = partition_images(coco, 3, proportions, seed=456)
        second = partition_images(coco, 3, proportions, seed=456)
        self.assertEqual(first, second)

        flat = [image_id for client in first for image_id in client]
        expected = {image["id"] for image in coco["images"]}
        self.assertEqual(set(flat), expected)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)

        anns_by_image = build_image_ann_map(coco)
        owner = {image_id: client_id for client_id, ids in enumerate(first) for image_id in ids}
        for image_id, annotations in anns_by_image.items():
            self.assertIn(image_id, owner)
            # Atomicity is an image-level invariant: all annotations resolve to
            # the one owner selected for their image.
            self.assertTrue(all(owner[ann["image_id"]] == owner[image_id] for ann in annotations))

        stats = split_statistics(coco, first)
        self.assertEqual(sum(row["num_images"] for row in stats), len(coco["images"]))
        self.assertEqual(
            sum(row["background_images"] for row in stats),
            sum(1 for image in coco["images"] if image["id"] not in anns_by_image),
        )

    def test_partition_rejects_malformed_proportions(self):
        coco = _toy_coco(6)
        cat_ids = sorted(category_mapping(coco))
        proportions = draw_client_proportions(cat_ids, 3, "iid", alpha=1.0, seed=1)
        proportions[str(cat_ids[0])] = [0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "Invalid proportions"):
            partition_images(coco, 3, proportions, seed=1)


if __name__ == "__main__":
    unittest.main()
