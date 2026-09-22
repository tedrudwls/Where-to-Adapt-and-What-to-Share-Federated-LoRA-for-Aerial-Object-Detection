"""CPU-only tests for prediction-blind qualitative-test selection and gates."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_test_qualitative.py"
SPEC = importlib.util.spec_from_file_location("render_test_qualitative", MODULE_PATH)
qual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qual)


class SourceDisjointSelectionTests(unittest.TestCase):
    def test_transitive_source_components_exclude_test(self):
        digest_a = "a" * 64
        digest_b = "b" * 64
        digest_c = "c" * 64
        records = {
            "train": {1: {"file_name": "same.rf." + "1" * 20 + ".jpg", "sha256": digest_a}},
            "val": {2: {"file_name": "same.rf." + "2" * 20 + ".jpg", "sha256": digest_b}},
            "test": {
                3: {"file_name": "other.rf." + "3" * 20 + ".jpg", "sha256": digest_b},
                4: {"file_name": "clean.rf." + "4" * 20 + ".jpg", "sha256": digest_c},
            },
        }
        # train -> val by Roboflow source key; val -> test 3 by exact SHA.
        self.assertEqual(qual.source_disjoint_test_ids(records), {4})

    def test_gt_only_median_selection_and_zoom(self):
        images = [{"id": i, "file_name": f"{i}.jpg", "width": 1024, "height": 1024}
                  for i in (1, 2, 3, 4, 5, 6)]
        coco = {
            "categories": [{"id": 3, "name": "drone"},
                           {"id": 4, "name": "helicopter"}],
            "images": images,
            "annotations": [
                {"image_id": 1, "category_id": 3, "bbox": [10, 10, 10, 10]},
                {"image_id": 2, "category_id": 3, "bbox": [20, 20, 20, 20]},
                {"image_id": 3, "category_id": 3, "bbox": [30, 30, 30, 30]},
                {"image_id": 4, "category_id": 4, "bbox": [40, 40, 10, 10]},
                {"image_id": 5, "category_id": 4, "bbox": [50, 50, 20, 20]},
                {"image_id": 6, "category_id": 4, "bbox": [60, 60, 30, 30]},
            ],
        }
        owner = {i: i % 3 for i in range(1, 7)}
        records = {i: {"sha256": f"{i}" * 64} for i in range(1, 7)}
        actual = qual.select_gt_examples(coco, set(owner), owner, records)
        self.assertEqual([item["image_id"] for item in actual], [2, 5])
        self.assertEqual([item["target_class"] for item in actual], ["drone", "helicopter"])
        for item in actual:
            x1, y1, x2, y2 = item["zoom_xyxy"]
            self.assertEqual(x2 - x1, y2 - y1)
            self.assertGreaterEqual(x1, 0)
            self.assertLessEqual(x2, 1024)

    def test_missing_positive_fails_instead_of_substitution(self):
        coco = {
            "categories": [{"id": 3, "name": "drone"},
                           {"id": 4, "name": "helicopter"}],
            "images": [{"id": 1, "file_name": "1.jpg", "width": 100, "height": 100}],
            "annotations": [{"image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10]}],
        }
        with self.assertRaisesRegex(ValueError, "helicopter"):
            qual.select_gt_examples(coco, {1}, {1: 0}, {1: {"sha256": "a" * 64}})

    def test_no_overwrite_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                qual._new_output_dir(Path(root))
            with self.assertRaises(ValueError):
                qual.safe_relative_name("../other.jpg")
            with self.assertRaises(ValueError):
                qual.safe_relative_name("/tmp/other.jpg")

    def test_prediction_boxes_keep_fixed_image_and_threshold(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values

        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "test.jpg"
            image.write_bytes(b"test")
            boxes = SimpleNamespace(xyxy=FakeTensor([[1, 2, 20, 30]]),
                                    conf=FakeTensor([0.8]), cls=FakeTensor([2.0]))
            result = SimpleNamespace(path=str(image), orig_shape=(64, 64), boxes=boxes)
            extracted = qual._extract_boxes(result, image, 64, 64)
            self.assertEqual(extracted[0]["class_id"], 2)
            self.assertEqual(extracted[0]["confidence"], 0.8)
            result.path = str(Path(temporary) / "different.jpg")
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                qual._extract_boxes(result, image, 64, 64)

    def test_checkpoint_index_requires_best_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = {"records": []}
            with self.assertRaisesRegex(ValueError, "Exactly one indexed checkpoint"):
                qual._indexed_methods(index, root, root, "a" * 64, "b" * 64)

    def test_indexed_methods_resolve_seed_directory_and_best_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            results_root = project / "results" / "official_v6"
            split_digest, weights_digest = "a" * 64, "b" * 64
            records = []
            for _, name, method, backbone, decoder in qual.METHODS:
                exp = f"seed_42/{name}"
                folder = results_root / exp
                weights = folder / "weights"
                weights.mkdir(parents=True)
                checkpoint = weights / "best_federated.pt"
                checkpoint.write_bytes(f"best:{name}".encode())
                result = {
                    "status": "complete", "mode": "fl", "fl_method": method,
                    "seed": 42, "partition_seed": 42,
                    "split_manifest_sha256": split_digest,
                    "lora_rank": 8, "apply_lora_backbone": backbone,
                    "apply_lora_decoder": decoder,
                    "selection": {"criterion": "best_macro_client_local_val_AP",
                                  "round": 20, "checkpoint": str(checkpoint)},
                }
                (folder / "fl_results.json").write_text(json.dumps(result), encoding="utf-8")
                records.append({
                    "experiment_id": exp, "mode": "fl", "method": method,
                    "partition": "dirichlet", "partition_seed": 42,
                    "training_seed": 42, "rank": 8,
                    "split_manifest_sha256": split_digest,
                    "pretrained_sha256": weights_digest,
                    "historical_project_relative_path": str(checkpoint.relative_to(project)),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": qual.file_sha256(checkpoint), "selected_at": 20,
                })
            selected = qual._indexed_methods({"records": records}, results_root,
                                              project, split_digest, weights_digest)
            self.assertEqual(len(selected), 5)
            self.assertTrue(all("seed_42" in str(item["result_path"]) for item in selected))

    def test_complete_select_stage_uses_only_coco_and_source_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "aod4"
            coco = {}
            rows = {}
            categories = [
                {"id": 1, "name": "airplane"}, {"id": 2, "name": "bird"},
                {"id": 3, "name": "drone"}, {"id": 4, "name": "helicopter"},
            ]
            for split, ids in (("train", (11,)), ("val", (12,)),
                               ("test", (1, 2, 3, 4, 5, 6))):
                folder = data_root / split
                folder.mkdir(parents=True)
                images, anns, inventory_rows = [], [], []
                for image_id in ids:
                    name = f"image_{image_id}.jpg"
                    content = f"synthetic-image-{image_id}".encode()
                    (folder / name).write_bytes(content)
                    images.append({"id": image_id, "file_name": name,
                                   "width": 1024, "height": 1024})
                    inventory_rows.append([image_id, name, qual.file_sha256(folder / name)])
                    if split == "test":
                        anns.append({"image_id": image_id,
                                     "category_id": 3 if image_id <= 3 else 4,
                                     "bbox": [10, 10, image_id * 10, image_id * 10]})
                coco[split] = {"images": images, "annotations": anns,
                               "categories": categories}
                annotation = folder / "_annotations.coco.json"
                annotation.write_text(json.dumps(coco[split]), encoding="utf-8")
                rows[split] = inventory_rows
            inventory = {
                "schema_version": 1,
                "identity_policy": qual.SOURCE_IDENTITY_POLICY,
                "records": rows,
                "per_split_image_tree_sha256": {
                    split: qual.tree_digest_from_inventory_rows(rows[split])
                    for split in qual.SPLITS
                },
            }
            inventory["inventory_sha256"] = qual.canonical_sha256(inventory)
            manifest = {
                "metadata": {
                    "schema_version": 7, "source_split_policy": "official_aod4_v6",
                    "partition": "dirichlet", "dirichlet_alpha": 0.4,
                    "seed": 42, "num_clients": 3,
                    "class_names": ["airplane", "bird", "drone", "helicopter"],
                    "annotation_sha256": {
                        split: qual.file_sha256(data_root / split / "_annotations.coco.json")
                        for split in qual.SPLITS
                    },
                    "source_hash_inventory": inventory,
                    "cross_split_source_audit": {"audit_sha256": "b" * 64},
                },
                "clients": [
                    {"client_id": client_id,
                     "splits": {"test": {"image_ids": [client_id + 1, client_id + 4]}}}
                    for client_id in range(3)
                ],
            }
            split_file = root / "split.json"
            split_file.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "selection"
            qual.select(SimpleNamespace(data_root=data_root, split_file=split_file,
                                        output_dir=output_dir))
            selection = qual.load_json(output_dir / "selection.json")
            self.assertEqual([item["target_class"] for item in selection["images"]],
                             ["drone", "helicopter"])
            self.assertEqual(selection["eligible_source_disjoint_test_images"], 6)
            self.assertTrue(all(item["source_group_id"] for item in selection["images"]))
            with self.assertRaises(FileExistsError):
                qual.select(SimpleNamespace(data_root=data_root, split_file=split_file,
                                            output_dir=output_dir))


if __name__ == "__main__":
    unittest.main()
