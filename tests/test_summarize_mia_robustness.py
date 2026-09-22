"""Tests for frozen-protocol, replicate-level MIA aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mia_robustness import (
    EXPERIMENT_NAMES,
    EXPECTED_AUGMENTATION_PROTOCOL,
    FROZEN_PROTOCOL,
    METHODS,
    METRICS,
    SCORES,
    SCOPES,
    aggregate_reports,
    audit_directory_name,
    load_reports,
    main,
    sha256_file,
    validate_report,
)


def _report(seed: int, base: float) -> dict:
    methods = {}
    for method_index, method in enumerate(METHODS):
        macro = {}
        for scope in SCOPES:
            macro[scope] = {}
            for score in SCORES:
                macro[scope][score] = {}
                for metric in METRICS:
                    value = base + method_index * 0.01
                    macro[scope][score][metric] = {
                        "client_macro_mean": value,
                        "client_sample_sd": 0.02,
                    }
        methods[method] = {"macro": macro}
        if seed != 42:
            methods[method]["primary_training_protocol_sha256"] = (
                f"{method_index + 1:064x}"
            )
    return {
        "audit_schema_version": 1,
        "status": "complete",
        "audit_name": "security_audit_v1",
        "audit_instance": audit_directory_name(seed),
        "seed": seed,
        "split_manifest_sha256": f"{seed:064x}",
        "model_weight_sha256": "a" * 64,
        "read_only_primary_gate": "pass",
        "exact_initial_digest_gate": "pass",
        "checkpoint_primary_architecture_manifest_gate": "pass",
        "frozen_primary_training_protocol_gate": "pass",
        "post_extraction_initial_model_state_gate": "pass",
        "sample_pairing_across_methods_gate": "pass",
        "repeated_attack_plan_pairing_gate": "pass",
        "protocol": dict(FROZEN_PROTOCOL),
        "methods": methods,
    }


def _primary_payload(seed: int, method: str) -> dict:
    uses_lora = method != "full_ft"
    training = {
        "fl_method": method,
        "model_name": "rtdetr-l",
        "model_weights": "/server/rtdetr-l.pt",
        "num_clients": 3,
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "apply_lora_backbone": True,
        "apply_lora_decoder": True,
        "backbone_min_channels": 64,
        "fl_rounds": 20,
        "local_epochs": 5,
        "batch_size": 8,
        "img_size": 640,
        "lr": 0.0003 if uses_lora else 0.0001,
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
        "val_interval": 5,
        "seed": seed,
        "partition_seed": seed,
        "num_workers": 4,
        "cross_client_eval": True,
        "visualize_interval": 5,
        "vis_samples": 6,
        "mia_max_samples": 1000,
        "mia_calibration_fraction": 0.5,
        "optimizer": "AdamW",
        "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
        "client_participation": "all_clients_every_round",
        "aggregation_weighting": "local_train_image_count",
        "nonfloating_state_policy": "server_keeps_existing_value",
        "validation_frequency_rounds": 1,
        "selection_criterion": "macro_client_local_validation_AP",
        "local_epoch_budget_per_client": 100,
        "communication_round_convention": (
            "num_clients_uploads_plus_num_clients_post_aggregation_downloads"
        ),
        "augmentation_protocol": EXPECTED_AUGMENTATION_PROTOCOL,
        "effective_close_mosaic_epochs": 10,
        "mia_attack": "image_level_ground_truth_matched_detection_loss_threshold",
        "mia_nonmember_source_policy": (
            "exclude_test_source_components_present_in_any_train_client"
        ),
        "mia_bootstrap": {
            "method": "stratified_nonparametric_full_attack_pipeline_percentile",
            "resamples": 1000,
            "confidence_level": 0.95,
            "recalibrates_direction_and_threshold": True,
        },
    }
    return {
        "status": "complete",
        "mode": "fl",
        "fl_method": method,
        "seed": seed,
        "partition_seed": seed,
        "partition": "dirichlet",
        "num_clients": 3,
        "rounds_executed": 20,
        "local_epochs": 5,
        "lora_rank": 8 if uses_lora else None,
        "lora_alpha": 16.0 if uses_lora else None,
        "apply_lora_backbone": True if uses_lora else None,
        "apply_lora_decoder": True if uses_lora else None,
        "architecture": {
            "model_name": "rtdetr-l",
            "num_classes": 4,
            "class_names": ["airplane", "bird", "drone", "helicopter"],
        },
        "training_experiment": training,
    }


def _write_audit(root: Path, seed: int, base: float) -> dict:
    directory = root / audit_directory_name(seed)
    directory.mkdir(parents=True)
    report = _report(seed, base)
    # Exercise backward compatibility with the already-completed seed-42 report.
    if seed == 42:
        report.pop("audit_instance")
    report_path = directory / "audit_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    protected = {}
    for method in METHODS:
        primary_path = (
            root / f"seed_{seed}" / EXPERIMENT_NAMES[method] / "fl_results.json"
        )
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        primary_path.write_text(
            json.dumps(_primary_payload(seed, method)), encoding="utf-8"
        )
        protected[str(primary_path)] = {
            "path": str(primary_path),
            "size_bytes": primary_path.stat().st_size,
            "sha256": sha256_file(primary_path),
        }
    integrity = {
        "schema_version": 1,
        "status": "pass",
        "read_only_primary_gate": True,
        "before": protected,
        "after": json.loads(json.dumps(protected)),
        "changes": [],
        "generated_yolo_tree_sha256_before": "b" * 64,
        "generated_yolo_tree_sha256_after": "b" * 64,
        "audit_outputs": {
            str(report_path): {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            }
        },
    }
    (directory / "integrity_manifest.json").write_text(
        json.dumps(integrity), encoding="utf-8"
    )
    return report


class FrozenMiaSummaryTests(unittest.TestCase):
    def test_aggregation_uses_three_audits_as_statistical_units(self):
        reports = {
            42: _report(42, 0.50),
            43: _report(43, 0.60),
            44: _report(44, 0.70),
        }
        rows, method_deltas, score_deltas = aggregate_reports(reports)
        target = next(
            row for row in rows
            if row["method"] == "full_ft"
            and row["scope"] == "local"
            and row["score"] == "trained_loss"
            and row["metric"] == "auc_roc"
        )
        self.assertEqual(target["n_paired_replicates"], 3)
        self.assertAlmostEqual(target["mean"], 0.60)
        self.assertAlmostEqual(target["replicate_sample_sd"], 0.10)
        self.assertTrue(method_deltas)
        self.assertTrue(score_deltas)

    def test_report_protocol_drift_is_rejected(self):
        report = _report(43, 0.5)
        report["protocol"]["attack_repeats"] = 19
        with self.assertRaisesRegex(ValueError, "Frozen protocol mismatch"):
            validate_report(
                report, seed=43,
                report_path=Path("security_audit_v1_seed43/audit_report.json"),
            )

    def test_seed43_requires_exact_audit_instance(self):
        report = _report(43, 0.5)
        report.pop("audit_instance")
        with self.assertRaisesRegex(ValueError, "Audit instance mismatch"):
            validate_report(
                report, seed=43,
                report_path=Path("security_audit_v1_seed43/audit_report.json"),
            )

    def test_load_requires_integrity_bound_reports_and_seed42_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
                _write_audit(root, seed, base)
            reports, provenance = load_reports(root)
            self.assertEqual(tuple(sorted(reports)), (42, 43, 44))
            self.assertEqual(len(provenance), 3)

            report_path = root / audit_directory_name(44) / "audit_report.json"
            report_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after"):
                load_reports(root)

    def test_current_primary_drift_is_rejected_even_with_stale_pass_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
                _write_audit(root, seed, base)
            primary = (
                root / "seed_43" / EXPERIMENT_NAMES["lora"] / "fl_results.json"
            )
            primary.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input (size|hash) changed"):
                load_reports(root)

    def test_internally_inconsistent_integrity_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
                _write_audit(root, seed, base)
            manifest_path = (
                root / audit_directory_name(43) / "integrity_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["changes"] = [{"path": "/primary"}]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "records protected-file changes"):
                load_reports(root)

    def test_cross_seed_primary_protocol_fingerprint_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
                _write_audit(root, seed, base)
            primary = (
                root / "seed_44" / EXPERIMENT_NAMES["lora"] / "fl_results.json"
            )
            payload = json.loads(primary.read_text(encoding="utf-8"))
            payload["training_experiment"]["unrecognized_protocol_revision"] = 2
            primary.write_text(json.dumps(payload), encoding="utf-8")

            manifest_path = (
                root / audit_directory_name(44) / "integrity_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = {
                "path": str(primary),
                "size_bytes": primary.stat().st_size,
                "sha256": sha256_file(primary),
            }
            manifest["before"][str(primary)] = record
            manifest["after"][str(primary)] = json.loads(json.dumps(record))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol drift across seeds"):
                load_reports(root)

    def test_only_frozen_seed_set_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "exactly seeds"):
                load_reports(root, (42, 43))

    def test_cli_writes_only_dedicated_summary_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
                _write_audit(root, seed, base)
            self.assertEqual(main(["--results_root", str(root)]), 0)
            output = root / "security_audit_v1_multiseed_summary"
            expected = {
                "summary_long.csv",
                "paired_method_differences.csv",
                "paired_trained_delta_differences.csv",
                "summary.md",
                "summary.json",
                "summary_manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertIn(
                "three paired replicates",
                (output / "summary.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for name in expected:
                self.assertEqual((output / name).stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
