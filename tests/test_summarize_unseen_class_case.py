"""Regression tests for the fixed client-local unseen-class case extractor."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_unseen_class_case import (
    EXPECTED_CLASS_NAMES,
    METHODS,
    NUM_CLIENTS,
    SEEDS,
    result_path,
    run,
    split_path,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnseenClassCaseSummaryTests(unittest.TestCase):
    method_ap = {
        "local_full_ft": 0.10,
        "local_lora": 0.12,
        "centralized_full_ft": 0.70,
        "centralized_lora": 0.65,
        "fl_full_ft": 0.60,
        "fl_lora": 0.62,
        "fedsa_lora": 0.40,
        "fixed_share_b_lora": 0.45,
    }

    @staticmethod
    def _split_row(seed: int, client_id: int, split: str) -> dict:
        class_instances = {"1": 20, "2": 21, "3": 22, "4": 100}
        class_images = {"1": 15, "2": 16, "3": 17, "4": 80}
        if split == "train" and seed == 43:
            helicopter_boxes = (3452, 0, 2076)[client_id]
            helicopter_images = (2500, 0, 1500)[client_id]
            class_instances["4"] = helicopter_boxes
            class_images["4"] = helicopter_images
        elif split == "test":
            class_instances["4"] = (200, 300, 287)[client_id]
            class_images["4"] = (150, 220, 210)[client_id]
        elif split == "val" and seed == 43 and client_id == 1:
            class_instances["4"] = 23
            class_images["4"] = 17
        return {
            "client_id": client_id,
            "num_images": 100,
            "class_instances": class_instances,
            "class_images": class_images,
            "background_images": 0,
        }

    def _manifest(self, seed: int) -> dict:
        stats = {
            split: [
                self._split_row(seed, client_id, split)
                for client_id in range(NUM_CLIENTS)
            ]
            for split in ("train", "val", "test")
        }
        metadata = {
            "schema_version": 7,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "num_clients": 3,
            "seed": seed,
            "source_split_policy": "official_aod4_v6",
            "official_split_preserved": True,
            "client_partition_unit": "source_group",
            "min_bbox_area": 0.0,
            "min_bbox_side": 0.0,
            "drop_empty_images": False,
            "crowd_policy": (
                "require_zero_crowd_annotations_for_YOLO_metric_equivalence"
            ),
            "category_policy": (
                "exact_aod4_targets_ignore_only_unreferenced_declared_categories"
            ),
            "class_names": list(EXPECTED_CLASS_NAMES),
            "cat_id_to_label": {"1": 0, "2": 1, "3": 2, "4": 3},
            "realized_partition_statistics": stats,
            "fixture_marker": f"seed-{seed}",
        }
        clients = []
        for client_id in range(NUM_CLIENTS):
            clients.append(
                {
                    "client_id": client_id,
                    "splits": {
                        split: {
                            key: copy.deepcopy(value)
                            for key, value in stats[split][client_id].items()
                            if key != "client_id"
                        }
                        for split in ("train", "val", "test")
                    },
                }
            )
        return {"metadata": metadata, "clients": clients}

    @staticmethod
    def _metric_block(ap: float, *, support: int = 787) -> dict:
        return {
            "AP": ap + 0.01,
            "AP50": ap + 0.11,
            "AP75": ap + 0.06,
            "class_support": {"helicopter": support},
            "per_class": {
                "helicopter": {
                    "AP": ap,
                    "AP50": ap + 0.10,
                    "AP75": ap + 0.05,
                    "support": support,
                }
            },
        }

    def _payload(
        self,
        spec,
        seed: int,
        split_digest: str,
        metadata: dict,
        client_id: int | None,
    ) -> dict:
        lr = 0.0001 if spec.fl_method == "full_ft" else 0.0003
        training_experiment = {
            "fl_method": spec.fl_method,
            "model_name": "rtdetr-l",
            "num_clients": 3,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "batch_size": 8,
            "img_size": 640,
            "lr": lr,
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
            "seed": seed,
            "partition_seed": seed,
        }
        if spec.fl_method != "full_ft":
            training_experiment.update(
                {
                    "lora_rank": 8,
                    "lora_alpha": 16.0,
                    "lora_dropout": 0.0,
                    "apply_lora_backbone": True,
                    "apply_lora_decoder": True,
                    "backbone_min_channels": 64,
                }
            )
        if spec.mode == "fl":
            training_experiment.update(
                {
                    "fl_rounds": 20,
                    "local_epochs": 5,
                    "client_participation": "all_clients_every_round",
                    "aggregation_weighting": "local_train_image_count",
                    "nonfloating_state_policy": "retain_previous_server_value",
                    "validation_frequency_rounds": 1,
                    "selection_criterion": "macro_client_local_validation_AP",
                    "local_epoch_budget_per_client": 100,
                }
            )
        elif spec.mode == "solo":
            training_experiment.update(
                {
                    "solo_epochs": 100,
                    "checkpoint_selection": "single_client_validation_AP",
                }
            )
        else:
            training_experiment.update(
                {
                    "centralized_epochs": 100,
                    "checkpoint_selection": "macro_client_local_validation_AP",
                }
            )

        method_name = (
            f"Local-{spec.fl_method}"
            if spec.mode == "solo"
            else f"Centralized-{spec.fl_method}"
            if spec.mode == "centralized"
            else f"FL+{spec.fl_method}"
        )
        payload = {
            "result_schema_version": 2,
            "status": "complete",
            "mode": spec.mode,
            "method": method_name,
            "fl_method": spec.fl_method,
            "seed": seed,
            "partition_seed": seed,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "num_clients": 3,
            "lora_rank": 8 if spec.fl_method != "full_ft" else None,
            "lora_alpha": 16.0 if spec.fl_method != "full_ft" else None,
            "apply_lora_backbone": (
                True if spec.fl_method != "full_ft" else None
            ),
            "apply_lora_decoder": (
                True if spec.fl_method != "full_ft" else None
            ),
            "training_experiment": training_experiment,
            "split_metadata": copy.deepcopy(metadata),
            "architecture": {
                "model_name": "rtdetr-l",
                "num_classes": 4,
                "class_names": list(EXPECTED_CLASS_NAMES),
                "fine_tuning_mode": spec.fl_method,
                "ultralytics_version": "8.4.126",
                "model_weight_sha256": "a" * 64,
            },
        }
        if spec.mode == "fl":
            payload.update(
                {
                    "split_manifest_sha256": split_digest,
                    "rounds_planned": 20,
                    "rounds_executed": 20,
                    "local_epochs": 5,
                }
            )
            focal_ap = self.method_ap[spec.key]
            entries = []
            for endpoint_id in (2, 0, 1):
                ap = focal_ap if endpoint_id == 1 else 0.90 + endpoint_id * 0.01
                metrics = self._metric_block(ap)
                entries.append(
                    {"client_id": endpoint_id, **metrics, "metrics": metrics}
                )
            payload["common_test"] = {"per_client_model": entries}
        else:
            payload.update(
                {
                    "split_file_sha256": split_digest,
                    "epochs_budget": 100,
                    "training": {"epochs_executed": 100},
                    "common_test": self._metric_block(self.method_ap[spec.key]),
                }
            )
            if spec.mode == "solo":
                payload["client_id"] = client_id
        return payload

    def _write_fixture(self, root: Path) -> tuple[Path, Path, list[Path]]:
        results_root = root / "results" / "official_v6"
        split_dir = root / "data" / "splits"
        split_dir.mkdir(parents=True)
        protected = []
        for seed in SEEDS:
            manifest = self._manifest(seed)
            manifest_path = split_path(split_dir, seed)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            protected.append(manifest_path)
            split_digest = _sha256(manifest_path)
            for spec in METHODS:
                client_ids = range(NUM_CLIENTS) if spec.client_specific_file else (None,)
                for client_id in client_ids:
                    path = result_path(results_root, spec, seed, client_id)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = self._payload(
                        spec,
                        seed,
                        split_digest,
                        manifest["metadata"],
                        client_id,
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    protected.append(path)
        self.assertEqual(len(protected), 39)
        return results_root, split_dir, protected

    def test_exact_case_matrix_focal_endpoint_deltas_and_read_only_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, protected = self._write_fixture(root)
            before = {path: _sha256(path) for path in protected}
            output_dir = root / "summary"

            summary = run(results_root, split_dir, output_dir)

            self.assertEqual(summary["case_n"], 1)
            self.assertTrue(summary["descriptive_only"])
            self.assertEqual(
                (summary["case"]["seed"], summary["case"]["client_id"]),
                (43, 1),
            )
            self.assertEqual(summary["case"]["class_name"], "helicopter")
            self.assertEqual(summary["case"]["train_boxes"], 0)
            self.assertEqual(summary["case"]["other_client_train_boxes"], 5528)
            self.assertEqual(summary["case"]["common_test_boxes"], 787)
            self.assertEqual(len(summary["rows"]), 8)

            rows = {row["method_key"]: row for row in summary["rows"]}
            self.assertAlmostEqual(rows["fl_lora"]["AP"], 0.62)
            self.assertAlmostEqual(
                rows["fl_lora"]["delta_AP_vs_matched_local"], 0.50
            )
            self.assertEqual(
                rows["local_lora"][
                    "other_client_train_boxes_contributing_signal"
                ],
                0,
            )
            self.assertEqual(
                rows["fixed_share_b_lora"][
                    "other_client_train_boxes_contributing_signal"
                ],
                5528,
            )
            fixed_minus_fedsa = next(
                row
                for row in summary["comparisons"]
                if row["left_method_key"] == "fixed_share_b_lora"
                and row["right_method_key"] == "fedsa_lora"
            )
            self.assertAlmostEqual(fixed_minus_fedsa["delta_AP"], 0.05)

            for name in (
                "unseen_class_case.csv",
                "comparisons.csv",
                "summary.json",
                "summary.md",
                "summary_manifest.json",
            ):
                self.assertTrue((output_dir / name).is_file())
            integrity = json.loads(
                (output_dir / "summary_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(integrity["input_file_count"], 39)
            self.assertTrue(integrity["input_files_byte_identical"])
            self.assertEqual(before, {path: _sha256(path) for path in protected})
            markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("(`n=1`)", markdown)
            self.assertNotIn("±", markdown)

    def test_selection_is_training_only_and_rejects_a_second_zero_support_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, _ = self._write_fixture(root)
            path = split_path(split_dir, 42)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            client = manifest["clients"][0]
            client["splits"]["train"]["class_instances"]["1"] = 0
            client["splits"]["train"]["class_images"]["1"] = 0
            path.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(ValueError, "data-only primary zero-support"):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())

    def test_rejects_incomplete_or_nonprimary_results_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, _ = self._write_fixture(root)
            missing = result_path(results_root, METHODS[0], 42, 0)
            missing.unlink()
            output_dir = root / "must_not_exist"
            with self.assertRaises(FileNotFoundError):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, _ = self._write_fixture(root)
            spec = next(item for item in METHODS if item.key == "fl_lora")
            path = result_path(results_root, spec, 44)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rounds_executed"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "rounds_executed mismatch"):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())

    def test_rejects_wrong_support_and_stale_embedded_split_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, _ = self._write_fixture(root)
            spec = next(item for item in METHODS if item.key == "fl_lora")
            path = result_path(results_root, spec, 43)
            payload = json.loads(path.read_text(encoding="utf-8"))
            focal = next(
                entry
                for entry in payload["common_test"]["per_client_model"]
                if entry["client_id"] == 1
            )
            for block in (focal, focal["metrics"]):
                block["class_support"]["helicopter"] = 786
                block["per_class"]["helicopter"]["support"] = 786
            path.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "Common pooled-test support"):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir, _ = self._write_fixture(root)
            spec = next(item for item in METHODS if item.key == "centralized_lora")
            path = result_path(results_root, spec, 42)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["split_metadata"]["fixture_marker"] = "stale"
            path.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "embedded split_metadata differs"):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
