"""Tests for the single-checkpoint, read-only evaluation vertical slice."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate_checkpoint as target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _tree_snapshot(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = (
                _sha256(path),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_mode,
            )
    return result


class RepresentativeFixture:
    def __init__(self, root: Path):
        self.root = root
        self.data_root = root / "relocated" / "Images"
        self.test_root = self.data_root / "test"
        self.test_root.mkdir(parents=True)
        categories = [
            {"id": 0, "name": "airplane-helicopter-drone-bird"},
            {"id": 1, "name": "airplane"},
            {"id": 2, "name": "bird"},
            {"id": 3, "name": "drone"},
            {"id": 4, "name": "helicopter"},
        ]
        images, annotations, inventory = [], [], []
        for image_id in (1, 2, 3):
            file_name = f"image_{image_id}.jpg"
            image_path = self.test_root / file_name
            image_path.write_bytes(f"synthetic-image-{image_id}".encode("ascii"))
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": 100,
                "height": 80,
            })
            annotations.append({
                "id": image_id,
                "image_id": image_id,
                "category_id": image_id,
                "bbox": [10, 10, 20, 15],
                "area": 300,
                "iscrowd": 0,
            })
            inventory.append([image_id, file_name, _sha256(image_path)])
        annotation_payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        self.annotation = self.test_root / "_annotations.coco.json"
        _write_json(self.annotation, annotation_payload)

        class_names = ["airplane", "bird", "drone", "helicopter"]
        declared = {str(row["id"]): row["name"] for row in categories}
        raw_counts = {"0": 0, "1": 1, "2": 1, "3": 1, "4": 0}
        clients = []
        for client_id, image_id in enumerate((1, 2, 3)):
            train_count = 10 + client_id
            clients.append({
                "client_id": client_id,
                "splits": {
                    "train": {
                        "num_images": train_count,
                        "image_ids": list(range(100 * (client_id + 1),
                                                100 * (client_id + 1) + train_count)),
                    },
                    "val": {
                        "num_images": 2,
                        "image_ids": [400 + 2 * client_id, 401 + 2 * client_id],
                    },
                    "test": {"num_images": 1, "image_ids": [image_id]},
                },
            })
        split_counts = {
            "train": {"images": 33, "annotations": 0},
            "val": {"images": 6, "annotations": 0},
            "test": {"images": 3, "annotations": 3},
        }
        realized = {
            split_name: [
                {
                    "client_id": client_id,
                    "num_images": client["splits"][split_name]["num_images"],
                }
                for client_id, client in enumerate(clients)
            ]
            for split_name in ("train", "val", "test")
        }
        self.manifest_payload = {
            "metadata": {
                **copy.deepcopy(target.FROZEN_MANIFEST_PROTOCOL),
                "data_root": "/historical/server/AOD4/Images",
                "annotation_sha256": {"test": _sha256(self.annotation)},
                "split_counts": split_counts,
                "source_split_counts": copy.deepcopy(split_counts),
                "realized_partition_statistics": realized,
                "source_category_audit": {
                    "test": {
                        "declared_categories": declared,
                        "raw_annotation_counts": raw_counts,
                    }
                },
                "source_hash_inventory": {
                    "schema_version": 1,
                    "identity_policy": target.FROZEN_MANIFEST_PROTOCOL[
                        "source_identity_policy"
                    ],
                    "inventory_sha256": "a" * 64,
                    "records": {"test": inventory},
                    "per_split_image_tree_sha256": {
                        "test": target._inventory_tree_digest(inventory)
                    },
                },
            },
            "clients": clients,
        }
        self.split = root / "split.json"
        _write_json(self.split, self.manifest_payload)
        self.replay_payload = {
            "schema_version": 1,
            "protocol": "fedlora_representative_test_replay",
            "experiment_id": target.TARGET_EXPERIMENT_ID,
            "historical_split_manifest": {
                "schema_version": target.FROZEN_MANIFEST_PROTOCOL["schema_version"],
                "sha256": _sha256(self.split),
            },
            "partition_protocol": copy.deepcopy(target.FROZEN_MANIFEST_PROTOCOL),
            "split_counts": copy.deepcopy(split_counts),
            "client_image_counts": {
                split_name: [
                    client["splits"][split_name]["num_images"]
                    for client in clients
                ]
                for split_name in ("train", "val", "test")
            },
            "test": {
                "annotation": {
                    "relative_path": "test/_annotations.coco.json",
                    "sha256": _sha256(self.annotation),
                },
                "category_audit": copy.deepcopy(
                    self.manifest_payload["metadata"]["source_category_audit"]["test"]
                ),
                "source_inventory": {
                    "schema_version": 1,
                    "identity_policy": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["identity_policy"],
                    "historical_inventory_sha256": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["inventory_sha256"],
                    "image_tree_sha256": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["per_split_image_tree_sha256"]["test"],
                    "records": copy.deepcopy(inventory),
                },
                "client_image_ids": [
                    {
                        "client_id": client_id,
                        "image_ids": copy.deepcopy(client["splits"]["test"]["image_ids"]),
                    }
                    for client_id, client in enumerate(clients)
                ],
            },
        }
        self.replay = root / "replay.json"
        _write_json(self.replay, self.replay_payload)

        self.checkpoint = root / "best_federated.pt"
        self.checkpoint.write_bytes(b"synthetic-checkpoint")
        self.model_weights = root / "rtdetr-l.pt"
        self.model_weights.write_bytes(b"synthetic-pretrained-model")

        self.expected = {
            "client_local_test": [
                {"client_id": 0, "num_images": 1, "AP": 0.1, "AP50": 0.2, "AP75": 0.05},
                {"client_id": 1, "num_images": 1, "AP": 0.2, "AP50": 0.3, "AP75": 0.15},
                {"client_id": 2, "num_images": 1, "AP": 0.3, "AP50": 0.4, "AP75": 0.25},
            ],
            "client_macro": {"AP": 0.2, "AP50": 0.3, "AP75": 0.15},
            "common_per_client_model": [
                {"client_id": 0, "AP": 0.11, "AP50": 0.21, "AP75": 0.06},
                {"client_id": 1, "AP": 0.21, "AP50": 0.31, "AP75": 0.16},
                {"client_id": 2, "AP": 0.31, "AP50": 0.41, "AP75": 0.26},
            ],
            "common_macro": {"AP": 0.21, "AP50": 0.31, "AP75": 0.16},
        }
        archived = {
            "status": "complete",
            "mode": "fl",
            "fl_method": "fedsa_lora",
            "seed": 42,
            "partition_seed": 42,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "num_clients": 3,
            "rounds_planned": 20,
            "rounds_executed": 20,
            "local_epochs": 5,
            "federated_checkpoint_schema_version": target.CHECKPOINT_SCHEMA,
            "federated_payload_policy": target.FEDSA_PAYLOAD_POLICY,
            "shared_lora_factor_role": "A",
            "client_local_lora_factor_role": "B",
            "selection": {
                "criterion": target.FROZEN_CHECKPOINT_CONTRACT["selection"],
                "checkpoint": str(root / "weights" / "best_federated.pt"),
                "round": 20,
            },
            "client_local_test": copy.deepcopy(self.expected["client_local_test"]),
            "client_summary": {
                metric: {"macro_mean": self.expected["client_macro"][metric]}
                for metric in target.REPORT_METRICS
            },
            "common_test": {
                **copy.deepcopy(self.expected["common_macro"]),
                "per_client_model": copy.deepcopy(
                    self.expected["common_per_client_model"]
                ),
            },
        }
        self.archived_result = root / "fl_results.json"
        _write_json(self.archived_result, archived)

        self.record = {
            "experiment_id": target.TARGET_EXPERIMENT_ID,
            "mode": "fl",
            "method": "fedsa_lora",
            "partition": "dirichlet",
            "partition_seed": 42,
            "training_seed": 42,
            "rank": 8,
            "client_id": None,
            "selected_unit": "round",
            "selected_at": 20,
            "historical_project_relative_path": (
                "results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/"
                "weights/best_federated.pt"
            ),
            "release_asset_name": (
                "seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt"
            ),
            "bytes": self.checkpoint.stat().st_size,
            "sha256": _sha256(self.checkpoint),
            "pretrained_sha256": _sha256(self.model_weights),
            "split_manifest_sha256": _sha256(self.split),
        }
        self.index = root / "checkpoint_index.json"
        self.write_index()
        self.reference = root / "reference.json"
        self.write_reference()

    def write_index(self):
        _write_json(self.index, {
            "schema_version": 1,
            "record_count": 1,
            "total_bytes": self.record["bytes"],
            "records": [self.record],
        })

    def write_reference(self):
        _write_json(self.reference, {
            "schema_version": 1,
            "protocol": "read_only_representative_checkpoint_evaluation",
            "experiment_id": target.TARGET_EXPERIMENT_ID,
            "method": "fedsa_lora",
            "display_name": "FedLoRA-A (Share-A / local B)",
            "metric_scale": "0_to_1",
            "absolute_tolerance": 1e-6,
            "public_replay_manifest": {
                "file_name": (
                    "seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json"
                ),
                "bytes": self.replay.stat().st_size,
                "sha256": _sha256(self.replay),
            },
            "checkpoint": {
                "release_asset_name": self.record["release_asset_name"],
                "historical_project_relative_path": self.record[
                    "historical_project_relative_path"
                ],
                "bytes": self.record["bytes"],
                "sha256": self.record["sha256"],
                "selected_round": 20,
                "contract": copy.deepcopy(target.FROZEN_CHECKPOINT_CONTRACT),
                "compatibility": copy.deepcopy(target.FROZEN_COMPATIBILITY),
            },
            "pretrained_model": {
                "file_name": "rtdetr-l.pt",
                "sha256": self.record["pretrained_sha256"],
            },
            "split_manifest": {
                "file_name": "split.json",
                "sha256": self.record["split_manifest_sha256"],
                "protocol": copy.deepcopy(target.FROZEN_MANIFEST_PROTOCOL),
                "split_counts": copy.deepcopy(
                    self.manifest_payload["metadata"]["split_counts"]
                ),
                "client_image_counts": {
                    split_name: [
                        client["splits"][split_name]["num_images"]
                        for client in self.manifest_payload["clients"]
                    ]
                    for split_name in ("train", "val", "test")
                },
                "source_inventory": {
                    "identity_policy": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["identity_policy"],
                    "inventory_sha256": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["inventory_sha256"],
                    "test_image_tree_sha256": self.manifest_payload["metadata"]
                    ["source_hash_inventory"]["per_split_image_tree_sha256"]["test"],
                },
            },
            "archived_result": {
                "file_name": "fl_results.json",
                "bytes": self.archived_result.stat().st_size,
                "sha256": _sha256(self.archived_result),
            },
            "expected": self.expected,
        })

    def argv(self):
        return [
            "--checkpoint", str(self.checkpoint),
            "--checkpoint-index", str(self.index),
            "--reference", str(self.reference),
            "--reference-result", str(self.archived_result),
            "--model-weights", str(self.model_weights),
            "--split-file", str(self.split),
            "--data-root", str(self.data_root),
            "--device", "cpu",
        ]

    def public_argv(self):
        return [
            "--checkpoint", str(self.checkpoint),
            "--checkpoint-index", str(self.index),
            "--reference", str(self.reference),
            "--model-weights", str(self.model_weights),
            "--replay-manifest", str(self.replay),
            "--data-root", str(self.data_root),
            "--device", "cpu",
        ]

    def frozen_identity_patch(self):
        return mock.patch.multiple(
            target,
            FROZEN_CHECKPOINT_BYTES=self.record["bytes"],
            FROZEN_CHECKPOINT_SHA256=self.record["sha256"],
            FROZEN_PRETRAINED_SHA256=self.record["pretrained_sha256"],
            FROZEN_SPLIT_SHA256=self.record["split_manifest_sha256"],
            FROZEN_ARCHIVED_RESULT_BYTES=self.archived_result.stat().st_size,
            FROZEN_ARCHIVED_RESULT_SHA256=_sha256(self.archived_result),
            FROZEN_REFERENCE_SHA256=_sha256(self.reference),
        )


class ReadOnlyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = RepresentativeFixture(self.root)

    def invoke(self, evaluator):
        return self.invoke_argv(evaluator, self.fixture.argv())

    def invoke_argv(self, evaluator, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with self.fixture.frozen_identity_patch():
            with mock.patch.object(target, "_perform_model_evaluation", evaluator):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = target.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def good_evaluator(self, **_kwargs):
        print("synthetic model noise")
        return copy.deepcopy(self.fixture.expected)

    def test_relocated_evaluation_is_read_only_and_emits_one_json(self):
        before = _tree_snapshot(self.root)
        code, stdout, stderr = self.invoke(self.good_evaluator)
        after = _tree_snapshot(self.root)
        self.assertEqual(code, 0)
        self.assertEqual(before, after)
        report = json.loads(stdout)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["read_only"])
        self.assertEqual(report["metric_scale"], "0_to_1")
        self.assertEqual(report["inputs"]["test_images_verified"], 3)
        self.assertEqual(
            report["runtime"]["historical_data_root"],
            "/historical/server/AOD4/Images",
        )
        self.assertEqual(
            report["runtime"]["relocated_data_root_used"],
            str(self.fixture.data_root.resolve()),
        )
        self.assertIn("synthetic model noise", stderr)

    def test_public_replay_needs_no_historical_split_or_result(self):
        observed = {}

        def evaluator(**kwargs):
            observed["split_file"] = kwargs["split_file"]
            observed["split_sha256"] = kwargs["data_info"]["split_manifest_sha256"]
            observed["client_sizes"] = kwargs["data_info"]["client_sizes"]
            return copy.deepcopy(self.fixture.expected)

        before = _tree_snapshot(self.root)
        code, stdout, stderr = self.invoke_argv(
            evaluator, self.fixture.public_argv()
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(before, _tree_snapshot(self.root))
        report = json.loads(stdout)
        self.assertEqual(report["runtime"]["manifest_mode"], "public_replay")
        self.assertIsNone(report["runtime"]["historical_data_root"])
        self.assertIsNone(report["runtime"]["relocated_data_root_used"])
        self.assertFalse(report["inputs"]["archived_result_verified"])
        self.assertIsNone(report["inputs"]["archived_result"])
        self.assertEqual(
            report["inputs"]["historical_split_manifest_sha256"],
            self.fixture.record["split_manifest_sha256"],
        )
        self.assertEqual(
            report["inputs"]["public_replay_manifest"]["sha256"],
            _sha256(self.fixture.replay),
        )
        self.assertEqual(observed["split_file"], self.fixture.replay.resolve())
        self.assertEqual(
            observed["split_sha256"], self.fixture.record["split_manifest_sha256"]
        )
        self.assertEqual(observed["client_sizes"], [10, 11, 12])

    def test_public_replay_mutation_fails_before_model_evaluation(self):
        payload = copy.deepcopy(self.fixture.replay_payload)
        payload["test"]["client_image_ids"][0]["image_ids"] = [1, 2]
        _write_json(self.fixture.replay, payload)
        evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
        code, stdout, stderr = self.invoke_argv(
            evaluator, self.fixture.public_argv()
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("public replay manifest", stderr)
        evaluator.assert_not_called()

    def test_historical_mode_requires_archived_result(self):
        argv = self.fixture.argv()
        flag = argv.index("--reference-result")
        del argv[flag:flag + 2]
        evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
        code, stdout, stderr = self.invoke_argv(evaluator, argv)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("--reference-result is required", stderr)
        evaluator.assert_not_called()

    def test_fake_evaluator_receives_complete_temporary_yolo_layout(self):
        observed = {}

        def inspecting_evaluator(**kwargs):
            data_info = kwargs["data_info"]
            self.assertNotEqual(kwargs["checkpoint"], self.fixture.checkpoint)
            self.assertNotEqual(kwargs["model_weights"], self.fixture.model_weights)
            self.assertEqual(
                kwargs["checkpoint"].read_bytes(), self.fixture.checkpoint.read_bytes()
            )
            self.assertEqual(
                kwargs["model_weights"].read_bytes(),
                self.fixture.model_weights.read_bytes(),
            )
            self.assertEqual(data_info["class_names"], [
                "airplane", "bird", "drone", "helicopter",
            ])
            self.assertEqual(data_info["client_sizes"], [10, 11, 12])
            self.assertEqual(
                data_info["split_manifest_sha256"],
                self.fixture.record["split_manifest_sha256"],
            )
            yaml_paths = [Path(value) for value in data_info["client_yamls"]]
            yaml_paths.append(Path(data_info["full_yaml"]))
            observed["runtime_root"] = yaml_paths[0].parents[2]
            observed["layouts"] = []
            expected_ids = ((1,), (2,), (3,), (1, 2, 3))
            for yaml_path, image_ids in zip(yaml_paths, expected_ids):
                self.assertTrue(yaml_path.is_file())
                layout_root = yaml_path.parent
                yaml_text = yaml_path.read_text(encoding="utf-8")
                self.assertIn(
                    f"path: {json.dumps(str(layout_root.resolve()))}", yaml_text
                )
                self.assertIn("train: images\nval: images\ntest: images", yaml_text)
                self.assertIn("nc: 4", yaml_text)
                for class_id, class_name in enumerate((
                    "airplane", "bird", "drone", "helicopter",
                )):
                    self.assertIn(
                        f"  {class_id}: {json.dumps(class_name)}", yaml_text
                    )

                links = sorted((layout_root / "images").rglob("*.jpg"))
                labels = sorted((layout_root / "labels").rglob("*.txt"))
                self.assertEqual(
                    [path.name for path in links],
                    [f"image_{image_id}.jpg" for image_id in image_ids],
                )
                self.assertEqual(
                    [path.name for path in labels],
                    [f"image_{image_id}.txt" for image_id in image_ids],
                )
                for image_id, link, label in zip(image_ids, links, labels):
                    self.assertTrue(link.is_symlink())
                    source = (self.fixture.test_root / f"image_{image_id}.jpg").resolve()
                    self.assertEqual(Path(os.readlink(link)), source)
                    self.assertEqual(link.resolve(strict=True), source)
                    self.assertEqual(
                        label.read_text(encoding="utf-8"),
                        f"{image_id - 1} 0.20000000 0.21875000 "
                        "0.20000000 0.18750000\n",
                    )
                observed["layouts"].append({
                    "yaml": yaml_text,
                    "symlink_targets": [os.readlink(path) for path in links],
                    "labels": [path.read_text(encoding="utf-8") for path in labels],
                })
            return copy.deepcopy(self.fixture.expected)

        code, stdout, stderr = self.invoke(inspecting_evaluator)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "pass")
        self.assertEqual(len(observed["layouts"]), 4)
        # The evaluator can only see an ephemeral tree. It is removed before
        # the command returns, so no generated YAML/labels persist.
        self.assertFalse(observed["runtime_root"].exists())

    def test_model_evaluator_restores_personalized_checkpoint_before_test_metrics(self):
        compatibility = copy.deepcopy(target.FROZEN_COMPATIBILITY)
        payload = {
            "schema_version": target.CHECKPOINT_SCHEMA,
            "checkpoint_kind": target.CHECKPOINT_KIND,
            "fl_method": "fedsa_lora",
            "num_clients": 3,
            "round": 20,
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "federated_payload_policy": target.FEDSA_PAYLOAD_POLICY,
            "shared_lora_factor_role": "A",
            "local_lora_factor_role": "B",
            "class_names": ["airplane", "bird", "drone", "helicopter"],
            "client_sample_counts": [10, 11, 12],
            "shared_state": {"shared.A": object()},
            "local_personalized_states": [
                {f"client.{client_id}.B": object()} for client_id in range(3)
            ],
            "compatibility": compatibility,
            **copy.deepcopy(target.FROZEN_CHECKPOINT_CONTRACT),
        }
        data_info = {
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "class_names": ["airplane", "bird", "drone", "helicopter"],
            "client_sizes": [10, 11, 12],
        }
        models = [object(), object(), object()]
        call_order = []

        def make_models(args, received_data):
            call_order.append("new")
            self.assertIs(received_data, data_info)
            self.assertEqual(args.fl_method, "fedsa_lora")
            self.assertEqual(args.device, "cpu")
            return models

        def apply_checkpoint(received_payload, received_models, args, received_data):
            call_order.append("apply")
            self.assertIs(received_payload, payload)
            self.assertIs(received_models, models)
            self.assertIs(received_data, data_info)
            self.assertEqual(args.data_root, str(self.fixture.data_root))
            self.assertEqual(args.split_file, str(self.fixture.split))
            self.assertEqual(args.model_weights, str(self.fixture.model_weights))

        def evaluate_local(received_models, received_data, args, *, split):
            call_order.append("local")
            self.assertIs(received_models, models)
            self.assertIs(received_data, data_info)
            self.assertEqual(split, "test")
            rows = copy.deepcopy(self.fixture.expected["client_local_test"])
            summary = {
                metric: {"macro_mean": self.fixture.expected["client_macro"][metric]}
                for metric in target.REPORT_METRICS
            }
            return rows, summary

        def evaluate_common(received_models, received_data, args):
            call_order.append("common")
            self.assertIs(received_models, models)
            self.assertIs(received_data, data_info)
            return {
                **copy.deepcopy(self.fixture.expected["common_macro"]),
                "per_client_model": copy.deepcopy(
                    self.fixture.expected["common_per_client_model"]
                ),
            }

        fake_torch = types.ModuleType("torch")
        fake_torch.load = mock.Mock(return_value=payload)
        fake_server = types.ModuleType("trainers.fl_server")
        fake_server._new_client_models = mock.Mock(side_effect=make_models)
        fake_server._apply_checkpoint = mock.Mock(side_effect=apply_checkpoint)
        fake_server._evaluate_client_local = mock.Mock(side_effect=evaluate_local)
        fake_server._evaluate_common_test = mock.Mock(side_effect=evaluate_common)
        fake_trainers = types.ModuleType("trainers")
        fake_trainers.__path__ = []
        fake_trainers.fl_server = fake_server

        with mock.patch.dict(sys.modules, {
            "torch": fake_torch,
            "trainers": fake_trainers,
            "trainers.fl_server": fake_server,
        }):
            with mock.patch.object(target, "_set_seed") as set_seed:
                actual = target._perform_model_evaluation(
                    checkpoint=self.fixture.checkpoint,
                    record=self.fixture.record,
                    model_weights=self.fixture.model_weights,
                    split_file=self.fixture.split,
                    data_root=self.fixture.data_root,
                    data_info=data_info,
                    device="cpu",
                )

        self.assertEqual(actual, self.fixture.expected)
        self.assertEqual(call_order, ["new", "apply", "local", "common"])
        fake_torch.load.assert_called_once_with(
            str(self.fixture.checkpoint), map_location="cpu", weights_only=True
        )
        set_seed.assert_called_once_with(42)

    def test_checkpoint_mismatch_precedes_evaluator(self):
        self.fixture.checkpoint.write_bytes(b"X" * self.fixture.record["bytes"])
        evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
        code, stdout, stderr = self.invoke(evaluator)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("checkpoint SHA-256 mismatch", stderr)
        evaluator.assert_not_called()

    def test_split_and_pretrained_mismatches_precede_evaluator(self):
        for field, path, message in (
            ("split", self.fixture.split, "split manifest SHA-256 mismatch"),
            ("model", self.fixture.model_weights, "pretrained model SHA-256 mismatch"),
        ):
            with self.subTest(field=field):
                original = path.read_bytes()
                path.write_bytes(original + b"x")
                evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
                code, stdout, stderr = self.invoke(evaluator)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn(message, stderr)
                evaluator.assert_not_called()
                path.write_bytes(original)

    def test_relocated_image_content_is_verified(self):
        image = self.fixture.test_root / "image_2.jpg"
        image.write_bytes(b"tampered-image")
        evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
        code, stdout, stderr = self.invoke(evaluator)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Test image SHA-256 mismatch", stderr)
        evaluator.assert_not_called()

    def test_checkpoint_payload_contract_rejects_wrong_local_state_count(self):
        payload = {
            "schema_version": 5,
            "checkpoint_kind": "federated_personalized",
            "fl_method": "fedsa_lora",
            "num_clients": 3,
            "round": 20,
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "federated_payload_policy": target.FEDSA_PAYLOAD_POLICY,
            "shared_lora_factor_role": "A",
            "local_lora_factor_role": "B",
            "class_names": ["airplane", "bird", "drone", "helicopter"],
            "client_sample_counts": [10, 11, 12],
            "shared_state": {"tensor": object()},
            "local_personalized_states": [{"x": object()}, {"x": object()}],
            "compatibility": {},
        }
        data_info = {
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "class_names": ["airplane", "bird", "drone", "helicopter"],
            "client_sizes": [10, 11, 12],
        }
        with self.assertRaisesRegex(target.EvaluationError, "one local-B state"):
            target._validate_checkpoint_payload(payload, self.fixture.record, data_info)

    def test_checkpoint_payload_binds_protocol_and_validation_selection(self):
        data_info = {
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "class_names": ["airplane", "bird", "drone", "helicopter"],
            "client_sizes": [10, 11, 12],
        }
        payload = {
            **copy.deepcopy(target.FROZEN_CHECKPOINT_CONTRACT),
            "round": 20,
            "fl_method": "fedsa_lora",
            "num_clients": 3,
            "split_manifest_sha256": self.fixture.record["split_manifest_sha256"],
            "class_names": copy.deepcopy(data_info["class_names"]),
            "client_sample_counts": [10, 11, 12],
            "shared_state": {"shared.A": object()},
            "local_personalized_states": [
                {f"client.{client_id}.B": object()} for client_id in range(3)
            ],
            "compatibility": copy.deepcopy(target.FROZEN_COMPATIBILITY),
        }
        with self.subTest("rank"):
            changed = copy.deepcopy(payload)
            changed["compatibility"]["lora_rank"] = 16
            with self.assertRaisesRegex(target.EvaluationError, "compatibility mismatch"):
                target._validate_checkpoint_payload(
                    changed, self.fixture.record, data_info
                )
        with self.subTest("selection"):
            changed = copy.deepcopy(payload)
            changed["selection"] = "last_executed_round"
            with self.assertRaisesRegex(target.EvaluationError, "selection/contract"):
                target._validate_checkpoint_payload(
                    changed, self.fixture.record, data_info
                )

    def test_runtime_input_mutation_is_caught_and_emits_no_success_json(self):
        def mutating_evaluator(**_kwargs):
            self.fixture.checkpoint.write_bytes(b"mutated")
            return copy.deepcopy(self.fixture.expected)

        code, stdout, stderr = self.invoke(mutating_evaluator)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Protected input changed", stderr)

    def test_validation_window_checkpoint_replacement_is_caught(self):
        original_validate = target._validate_relocated_test_data
        replacement = b"X" * self.fixture.checkpoint.stat().st_size

        def validate_then_replace(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            self.fixture.checkpoint.write_bytes(replacement)
            return result

        evaluator = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(
            target, "_validate_relocated_test_data", side_effect=validate_then_replace
        ):
            code, stdout, stderr = self.invoke(evaluator)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Protected input changed", stderr)
        evaluator.assert_not_called()

    def test_post_staging_source_swap_cannot_reach_deserializer(self):
        original_bytes = self.fixture.checkpoint.read_bytes()
        original_build = target._build_temporary_data_info
        observed = {}

        def build_then_replace(*args, **kwargs):
            result = original_build(*args, **kwargs)
            self.fixture.checkpoint.write_bytes(b"Z" * len(original_bytes))
            return result

        def inspecting_evaluator(**kwargs):
            observed["loaded_bytes"] = kwargs["checkpoint"].read_bytes()
            return copy.deepcopy(self.fixture.expected)

        with mock.patch.object(
            target, "_build_temporary_data_info", side_effect=build_then_replace
        ):
            code, stdout, stderr = self.invoke(inspecting_evaluator)
        self.assertEqual(observed["loaded_bytes"], original_bytes)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Protected input changed", stderr)

    def test_native_fd_stdout_noise_does_not_corrupt_json(self):
        def noisy_evaluator(**_kwargs):
            os.write(1, b"native evaluator noise\n")
            return copy.deepcopy(self.fixture.expected)

        with tempfile.TemporaryFile() as native_stderr:
            saved_stderr_fd = os.dup(2)
            try:
                os.dup2(native_stderr.fileno(), 2)
                code, stdout, stderr = self.invoke(noisy_evaluator)
            finally:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            native_stderr.seek(0)
            native_noise = native_stderr.read().decode("utf-8")

        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "pass")
        self.assertNotIn("native evaluator noise", stdout)
        self.assertIn("native evaluator noise", native_noise)

    def test_metric_drift_returns_machine_readable_failure(self):
        def drifted_evaluator(**_kwargs):
            output = copy.deepcopy(self.fixture.expected)
            output["common_macro"]["AP"] += 0.01
            return output

        code, stdout, _stderr = self.invoke(drifted_evaluator)
        self.assertEqual(code, 1)
        report = json.loads(stdout)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["reference_comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
