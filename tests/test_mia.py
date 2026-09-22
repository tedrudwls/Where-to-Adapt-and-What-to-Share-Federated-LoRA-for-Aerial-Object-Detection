"""Tests for the standalone, read-only MIA robustness audit."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.mia_robustness_audit import (
    EXPECTED_AUGMENTATION_PROTOCOL,
    EXPERIMENT_NAMES,
    _cache_load_or_extract,
    _merge_losses,
    _strict_primary_result,
    _validate_primary_checkpoint_compatibility,
    _validate_architecture_provenance,
    audit_output_name,
    changed_snapshot,
    json_sha256,
    repeated_attack,
    resolve_primary_inputs,
    sha256_file,
    snapshot_files,
    validate_output_location,
    validate_existing_resume_plan,
    validate_resume_state,
)


def _record(sample: int, source: int, loss: float) -> dict:
    return {
        "sample_id_sha256": f"{sample:064x}",
        "image_content_sha256": f"{sample + 10_000:064x}",
        "source_group_sha256": f"{source:064x}",
        "split": "train" if loss < 1.0 else "test",
        "width": 100,
        "height": 100,
        "object_count": 1,
        "background": False,
        "class_counts": {"drone": 1},
        "classes": ["drone"],
        "bbox_area_ratio_mean": 0.01,
        "bbox_area_ratio_min": 0.01,
        "bbox_area_ratio_max": 0.01,
        "loss": float(loss),
        "loss_components": {},
    }


class IntegrityTests(unittest.TestCase):
    def test_architecture_provenance_requires_exact_manifest(self):
        architecture = {
            "model_weight_sha256": "b" * 64,
            "initial_target_tensor_state_sha256": "c" * 64,
            "federated_payload_policy": "all_trainable",
            "adapters": [{"name": "decoder.0", "rank": 8}],
            "parameter_counts": {"trainable_params": 123},
        }
        initial = _validate_architecture_provenance(
            method="lora",
            checkpoint_architecture=architecture,
            primary_architecture=json.loads(json.dumps(architecture)),
            model_weight_sha256="b" * 64,
        )
        self.assertEqual(initial, "c" * 64)
        drifted = json.loads(json.dumps(architecture))
        drifted["adapters"][0]["rank"] = 4
        with self.assertRaisesRegex(ValueError, "architecture manifests differ"):
            _validate_architecture_provenance(
                method="lora",
                checkpoint_architecture=architecture,
                primary_architecture=drifted,
                model_weight_sha256="b" * 64,
            )

    def test_primary_result_best_selection_tuple_is_validated(self):
        payload = {
            "status": "complete", "mode": "fl", "fl_method": "lora",
            "seed": 42, "partition_seed": 42, "partition": "dirichlet",
            "num_clients": 3, "rounds_executed": 20, "local_epochs": 5,
            "split_manifest_sha256": "a" * 64, "dirichlet_alpha": 0.4,
            "selection": {
                "criterion": "best_macro_client_local_val_AP",
                "checkpoint": "/old/server/path/best_federated.pt",
                "round": 20,
            },
        }
        _strict_primary_result(
            payload, method="lora", seed=42, split_sha256="a" * 64,
            best_checkpoint="/current/server/path/best_federated.pt",
        )
        payload["selection"]["checkpoint"] = "last_federated.pt"
        with self.assertRaisesRegex(ValueError, "checkpoint.basename"):
            _strict_primary_result(
                payload, method="lora", seed=42, split_sha256="a" * 64,
                best_checkpoint="/current/server/path/best_federated.pt",
            )

    def test_snapshot_detects_a_byte_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primary.json"
            path.write_bytes(b"before")
            before = snapshot_files([path])
            self.assertEqual(before[str(path.resolve())]["sha256"], sha256_file(path))
            path.write_bytes(b"after")
            after = snapshot_files([path])
            changes = changed_snapshot(before, after)
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["path"], str(path.resolve()))

    def test_output_is_exact_sibling_and_not_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            primary = root / "seed_42" / "fl_lora_r8_a0.4"
            primary.mkdir(parents=True)
            expected = root / "security_audit_v1"
            self.assertEqual(
                validate_output_location(root, expected, [primary]), expected.resolve()
            )
            with self.assertRaisesRegex(ValueError, "writes only"):
                validate_output_location(root, root / "somewhere_else", [primary])

            seed43 = root / "security_audit_v1_seed43"
            self.assertEqual(
                validate_output_location(root, seed43, [primary], seed=43),
                seed43.resolve(),
            )
            seed44 = root / "security_audit_v1_seed44"
            self.assertEqual(
                validate_output_location(root, seed44, [primary], seed=44),
                seed44.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "supports seeds"):
                audit_output_name(45)

    def test_primary_resolution_requires_best_and_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = "lora"
            experiment = root / "seed_42" / EXPERIMENT_NAMES[method]
            (experiment / "weights").mkdir(parents=True)
            (experiment / "fl_results.json").write_text("{}", encoding="utf-8")
            (experiment / "weights" / "best_federated.pt").write_bytes(b"best")
            with self.assertRaises(FileNotFoundError):
                resolve_primary_inputs(root, 42, [method])
            (experiment / "weights" / "last_federated.pt").write_bytes(b"last")
            resolved = resolve_primary_inputs(root, 42, [method])
            self.assertEqual(
                resolved[method]["best_checkpoint"],
                str((experiment / "weights" / "best_federated.pt").resolve()),
            )

    def test_completed_audit_cannot_be_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "security_audit_v1"
            output.mkdir()
            (output / "audit_report.json").write_text(
                '{"status":"complete"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                validate_resume_state(output, resume=True, dry_run=False)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_resume_state(output, resume=False, dry_run=False)
            # Inspection remains possible without mutating the completed audit.
            validate_resume_state(output, resume=True, dry_run=True)

    def test_interrupted_resume_requires_exact_plan_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "security_audit_v1_seed43"
            output.mkdir()
            plan = {
                "audit_schema_version": 1,
                "mode": "execute",
                "read_only_primary": True,
                "seed": 43,
                "audit_instance": "security_audit_v1_seed43",
                "methods": list(EXPERIMENT_NAMES),
                "output_dir": str(output),
                "runtime_cache_root": str(output / "runtime_cache"),
                "protected_inputs": {"/primary": {"sha256": "a" * 64}},
                "protocol": {"attack_repeats": 20},
            }
            (output / "audit_plan.json").write_text(
                json.dumps({**plan, "status": "running", "run_nonce": "x"}),
                encoding="utf-8",
            )
            validate_existing_resume_plan(
                output, plan, resume=True, dry_run=False
            )
            drifted = json.loads(json.dumps(plan))
            drifted["protocol"]["attack_repeats"] = 19
            with self.assertRaisesRegex(ValueError, "plan/protected-input mismatch"):
                validate_existing_resume_plan(
                    output, drifted, resume=True, dry_run=False
                )

    def test_primary_checkpoint_compatibility_is_exact(self):
        compatibility = {
            "fl_method": "lora", "model_name": "rtdetr-l",
            "num_classes": 4, "num_clients": 3, "fl_rounds": 20,
            "local_epochs": 5, "batch_size": 8, "img_size": 640,
            "num_workers": 4, "lr": 0.0003, "head_lr": 0.0001,
            "backbone_lr_ratio": 0.1, "weight_decay": 0.0001,
            "warmup_epochs": 5.0, "min_lr_ratio": 0.01,
            "grad_clip_norm": 0.1, "close_mosaic_epochs": 10,
            "fedprox_mu": 0.0, "reset_optimizer_each_round": True,
            "amp": False, "augmentation_protocol": EXPECTED_AUGMENTATION_PROTOCOL,
            "seed": 43, "partition_seed": 43, "partition": "dirichlet",
            "dirichlet_alpha": 0.4, "lora_rank": 8, "lora_alpha": 16.0,
            "lora_dropout": 0.0, "apply_lora_backbone": True,
            "apply_lora_decoder": True, "backbone_min_channels": 64,
        }
        digest = _validate_primary_checkpoint_compatibility(
            compatibility, method="lora", seed=43
        )
        self.assertEqual(len(digest), 64)
        compatibility["apply_lora_backbone"] = False
        with self.assertRaisesRegex(ValueError, "frozen primary configuration"):
            _validate_primary_checkpoint_compatibility(
                compatibility, method="lora", seed=43
            )


class RecordAndAttackTests(unittest.TestCase):
    def test_exact_cache_is_reused_within_fresh_multimethod_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared_initial.json"
            key = {
                "checkpoint": "initial", "samples": "paired",
                "sample_ids_sha256": json_sha256(["a" * 64]),
                "sample_count": 1,
            }
            calls = []

            def extractor():
                calls.append(True)
                return [{"sample_id_sha256": "a" * 64, "loss": 1.0}]

            first = _cache_load_or_extract(
                path, key, resume=False, extractor=extractor
            )
            second = _cache_load_or_extract(
                path, key, resume=False, extractor=extractor
            )
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_merge_is_exactly_sample_paired_and_computes_delta(self):
        trained = [_record(1, 101, 0.2), _record(2, 102, 0.3)]
        initial = [_record(2, 102, 1.3), _record(1, 101, 1.2)]
        for row in initial:
            row["split"] = "train"
        merged = _merge_losses(trained, initial)
        self.assertEqual([row["sample_id_sha256"] for row in merged], [f"{1:064x}", f"{2:064x}"])
        self.assertAlmostEqual(merged[0]["delta_loss"], -1.0)
        self.assertAlmostEqual(merged[1]["delta_loss"], -1.0)
        altered = list(initial)
        altered[0] = {**altered[0], "sample_id_sha256": f"{999:064x}"}
        with self.assertRaisesRegex(ValueError, "identical samples"):
            _merge_losses(trained, altered)

    def test_repeated_attack_uses_fixed_calibration_operating_points(self):
        try:
            import numpy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as error:
            self.skipTest(f"numeric MIA dependencies unavailable: {error}")
        members = []
        nonmembers = []
        # One image per group makes the expected source-group partition explicit.
        for index in range(40):
            member = _record(index + 1, index + 1, 0.1 + index * 1e-4)
            member["trained_loss"] = member["loss"]
            member["initial_loss"] = 0.5
            member["delta_loss"] = member["trained_loss"] - 0.5
            members.append(member)
            nonmember = _record(index + 1001, index + 1001, 2.0 + index * 1e-4)
            nonmember["trained_loss"] = nonmember["loss"]
            nonmember["initial_loss"] = 0.5
            nonmember["delta_loss"] = nonmember["trained_loss"] - 0.5
            nonmembers.append(nonmember)

        result = repeated_attack(
            members, nonmembers, score_field="trained_loss", repeats=4,
            attack_seed=123, calibration_fraction=0.5,
        )
        self.assertAlmostEqual(result["summary"]["auc_roc"]["mean"], 1.0)
        self.assertEqual(result["member_nonmember_source_group_intersection"], 0)
        self.assertTrue(result["calibration_evaluation_source_group_disjoint"])
        self.assertEqual(len(result["repeat_plan_sha256"]), 64)
        for run in result["runs"]:
            self.assertEqual(len(run["split_plan_sha256"]), 64)
            for key in ("1pct", "5pct", "10pct"):
                point = run["operating_points"][key]
                self.assertIn("calibration_achieved_fpr", point)
                self.assertIn("achieved_fpr", point)
                self.assertIn("false_positive_count", point)
                self.assertIn("true_positive_count", point)


if __name__ == "__main__":
    unittest.main()
