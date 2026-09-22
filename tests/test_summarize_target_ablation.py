"""Synthetic regression tests for the strict target-ablation summarizer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_target_ablation import (
    BASE_EXPERIMENT_NAMES,
    EFFICIENCY_FIELDS,
    METHODS,
    SEEDS,
    SUMMARY_FIELDS,
    TARGET_FLAGS,
    TARGETS,
    collect_runs,
    paired_deltas,
    run,
    summarize_runs,
)


class TargetAblationSummaryTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        results_root = root / "results" / "official_v6"
        split_dir = root / "data" / "splits"
        split_dir.mkdir(parents=True)
        model_digest = "a" * 64

        for seed in SEEDS:
            split_file = (
                split_dir
                / f"split_official_v6_dirichlet_a0.4_c3_s{seed}.json"
            )
            split_file.write_text(
                json.dumps({"seed": seed, "fixture": "target-ablation"}),
                encoding="utf-8",
            )
            split_digest = hashlib.sha256(split_file.read_bytes()).hexdigest()

            for method_index, method in enumerate(METHODS):
                for target_index, target in enumerate(TARGETS):
                    experiment = BASE_EXPERIMENT_NAMES[method]
                    if target != "both":
                        experiment += f"_{target}"
                    result_dir = results_root / f"seed_{seed}" / experiment
                    result_dir.mkdir(parents=True)

                    backbone, decoder = TARGET_FLAGS[target]
                    seed_index = seed - SEEDS[0]
                    macro_ap = (
                        0.40
                        + 0.08 * method_index
                        + 0.02 * target_index
                        + 0.01 * seed_index
                    )
                    client_sd_ap = 0.09 - 0.01 * target_index + 0.002 * seed_index
                    communication_params = (
                        100_000 + 20_000 * method_index + 10_000 * target_index
                    )
                    one_way_mb = communication_params * 4 / 1_000_000.0
                    trainable_params = 200_000 + 25_000 * target_index

                    client_summary = {}
                    common_test = {}
                    for metric, increment in (
                        ("AP", 0.0),
                        ("AP50", 0.10),
                        ("AP75", 0.04),
                    ):
                        value = macro_ap + increment
                        client_sd = client_sd_ap + (0.005 if metric != "AP" else 0.0)
                        client_summary[metric] = {
                            "macro_mean": value,
                            "client_sample_sd": client_sd,
                            "sample_std": client_sd,
                            "worst_client": value - 0.05,
                        }
                        common_test[metric] = value - 0.01

                    training = {
                        "fl_method": method,
                        "model_name": "rtdetr-l",
                        "num_clients": 3,
                        "fl_rounds": 20,
                        "local_epochs": 5,
                        "partition": "dirichlet",
                        "dirichlet_alpha": 0.4,
                        "lora_rank": 8,
                        "lora_alpha": 16.0,
                        "lora_dropout": 0.0,
                        "apply_lora_backbone": backbone,
                        "apply_lora_decoder": decoder,
                        "backbone_min_channels": 64,
                        "batch_size": 8,
                        "img_size": 640,
                        "lr": 0.0003,
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
                        "cross_client_eval": True,
                        "optimizer": "AdamW",
                        "lr_schedule": (
                            "global_step_linear_warmup_then_cosine_decay"
                        ),
                        "client_participation": "all_clients_every_round",
                        "aggregation_weighting": "local_train_image_count",
                        "nonfloating_state_policy": (
                            "retain_previous_server_value"
                        ),
                        "validation_frequency_rounds": 1,
                        "selection_criterion": (
                            "macro_client_local_validation_AP"
                        ),
                        "local_epoch_budget_per_client": 100,
                    }
                    payload = {
                        "result_schema_version": 2,
                        "status": "complete",
                        "mode": "fl",
                        "method": f"FL+{method}",
                        "fl_method": method,
                        "seed": seed,
                        "partition_seed": seed,
                        "partition": "dirichlet",
                        "dirichlet_alpha": 0.4,
                        "lora_rank": 8,
                        "lora_alpha": 16.0,
                        "apply_lora_backbone": backbone,
                        "apply_lora_decoder": decoder,
                        "num_clients": 3,
                        "rounds_planned": 20,
                        "rounds_executed": 20,
                        "local_epochs": 5,
                        "training_experiment": training,
                        "architecture": {
                            "model_name": "rtdetr-l",
                            "fine_tuning_mode": method,
                            "ultralytics_version": "8.4.126",
                            "model_weight_sha256": model_digest,
                        },
                        "split_manifest_sha256": split_digest,
                        "split_metadata": {
                            "schema_version": 7,
                            "partition": "dirichlet",
                            "dirichlet_alpha": 0.4,
                            "num_clients": 3,
                            "source_split_policy": "official_aod4_v6",
                            "official_split_preserved": True,
                            "client_partition_unit": "source_group",
                        },
                        "client_summary": client_summary,
                        "common_test": common_test,
                        "parameter_counts": {
                            "trainable_params": trainable_params,
                            "communication_params": communication_params,
                            "parameter_saving_pct": (
                                100.0 - trainable_params / 330_000.0
                            ),
                        },
                        "communication": {
                            "unit": "decimal_MB_1e6_bytes",
                            "scope": "model_tensor_payload_only",
                            "num_clients": 3,
                            "rounds_executed": 20,
                            "one_client_one_way": {
                                "params": communication_params,
                                "mb": one_way_mb,
                            },
                            "round_total": {"mb": one_way_mb * 6},
                            "cumulative_total": {"mb": one_way_mb * 120},
                            "byte_saving_vs_full_ft_pct": (
                                100.0 - communication_params / 330_000.0
                            ),
                        },
                    }
                    (result_dir / "fl_results.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
        return results_root, split_dir

    def test_exact_matrix_summary_deltas_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir = self._write_fixture(root)
            representative = (
                results_root
                / "seed_42"
                / "fl_lora_r8_a0.4_decoder_only"
                / "fl_results.json"
            )
            before_digest = hashlib.sha256(representative.read_bytes()).hexdigest()

            rows = collect_runs(results_root, split_dir)
            self.assertEqual(len(rows), 27)
            summaries = summarize_runs(rows)
            self.assertEqual(len(summaries), 9)
            first = summaries[0]
            self.assertAlmostEqual(first["macro_AP_mean"], 0.41)
            self.assertAlmostEqual(first["macro_AP_run_sd"], 0.01)
            for field in EFFICIENCY_FIELDS:
                self.assertAlmostEqual(first[f"{field}_run_sd"], 0.0)

            deltas = paired_deltas(rows)
            self.assertEqual(len(deltas), 18 * len(SUMMARY_FIELDS))
            both_minus_decoder = next(
                row
                for row in deltas
                if row["comparison_axis"] == "target"
                and row["context"] == "FL LoRA"
                and row["comparison"] == "both_minus_decoder"
                and row["metric"] == "macro_AP"
            )
            self.assertAlmostEqual(both_minus_decoder["mean_delta"], 0.04)
            self.assertAlmostEqual(both_minus_decoder["run_sd"], 0.0)
            self.assertEqual(both_minus_decoder["wins_for_numerator"], 3)
            lower_is_better = next(
                row
                for row in deltas
                if row["comparison_axis"] == "target"
                and row["context"] == "FL LoRA"
                and row["comparison"] == "both_minus_decoder"
                and row["metric"] == "client_sd_AP"
            )
            self.assertLess(lower_is_better["mean_delta"], 0.0)
            self.assertEqual(lower_is_better["wins_for_numerator"], 3)

            output_dir = root / "summary"
            paths = run(results_root, split_dir, output_dir)
            self.assertEqual(set(paths), {"runs", "summary", "deltas", "markdown"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            with paths["summary"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("## Client-local performance", markdown)
            self.assertIn("## Common pooled-test performance", markdown)
            self.assertIn("## Parameter and communication efficiency", markdown)
            self.assertEqual(
                hashlib.sha256(representative.read_bytes()).hexdigest(),
                before_digest,
            )

    def test_rejects_protocol_mismatch_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir = self._write_fixture(root)
            bad_result = (
                results_root
                / "seed_43"
                / "fl_fedsa_lora_r8_a0.4_backbone_only"
                / "fl_results.json"
            )
            payload = json.loads(bad_result.read_text(encoding="utf-8"))
            payload["training_experiment"]["lr"] = 0.001
            bad_result.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(ValueError, "training_experiment.lr mismatch"):
                run(results_root, split_dir, output_dir)
            self.assertFalse(output_dir.exists())

    def test_rejects_missing_coordinate_and_split_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir = self._write_fixture(root)
            missing = (
                results_root
                / "seed_44"
                / "fl_fixed_share_b_lora_r8_a0.4_backbone_only"
                / "fl_results.json"
            )
            missing.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing canonical"):
                collect_runs(results_root, split_dir)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root, split_dir = self._write_fixture(root)
            bad_result = (
                results_root
                / "seed_42"
                / "fl_lora_r8_a0.4"
                / "fl_results.json"
            )
            payload = json.loads(bad_result.read_text(encoding="utf-8"))
            payload["split_manifest_sha256"] = "f" * 64
            bad_result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split digest mismatch"):
                collect_runs(results_root, split_dir)


if __name__ == "__main__":
    unittest.main()
