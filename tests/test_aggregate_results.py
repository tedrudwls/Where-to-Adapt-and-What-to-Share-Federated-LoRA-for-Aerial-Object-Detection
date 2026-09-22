"""Regression tests for paired publication-table construction."""

from __future__ import annotations

import unittest

from scripts.aggregate_results import (
    _sha256_of_json,
    _split_generation_protocol,
    _validate_result_envelope,
    paired_factor_sharing_comparisons,
)


class FactorSharingAggregationTests(unittest.TestCase):
    def _row(self, method: str, label: str) -> dict:
        protocol = {
            "fl_method": method,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
            "backbone_min_channels": 64,
            "lr": 3e-4,
            "head_lr": 1e-4,
            "backbone_lr_ratio": 0.1,
            "fl_rounds": 20,
            "local_epochs": 5,
            "optimizer": "AdamW",
        }
        row = {
            "mode": "fl",
            "fl_method": method,
            "method": label,
            "seed": 42,
            "partition_seed": 42,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "num_clients": 3,
            "lora_rank": 8,
            "lora_alpha": 16,
            "apply_lora_backbone": True,
            "apply_lora_decoder": True,
            "split_manifest_sha256": "a" * 64,
            "protocol_manifest": protocol,
            "path": f"/{method}.json",
        }
        for metric in (
            "macro_AP", "macro_AP50", "macro_AP75", "client_sd_AP",
            "client_sd_AP50", "client_sd_AP75", "worst_AP", "worst_AP50",
            "worst_AP75", "common_AP",
        ):
            row[metric] = 0.5 if method == "fedsa_lora" else 0.4
        row["communication_params"] = (
            100.0 if method == "fedsa_lora" else 120.0
        )
        row["one_client_one_way_mb"] = (
            1.0 if method == "fedsa_lora" else 1.2
        )
        row["cumulative_total_mb"] = (
            120.0 if method == "fedsa_lora" else 144.0
        )
        return row

    def test_pairs_only_identical_controls_and_reports_communication_delta(self):
        proposal = self._row("fedsa_lora", "FL+fedsa_lora")
        baseline = self._row(
            "fixed_share_b_lora", "FL+fixed_share_b_lora"
        )

        pairs, summaries = paired_factor_sharing_comparisons(
            [proposal, baseline]
        )

        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["delta_macro_AP"], 0.1)
        self.assertAlmostEqual(
            pairs[0]["delta_communication_params"], -20.0
        )
        communication = next(
            row for row in summaries
            if row["metric"] == "communication_params"
        )
        self.assertAlmostEqual(
            communication["mean_delta_proposal_minus_baseline"], -20.0
        )
        self.assertEqual(communication["preferred_direction"], "lower")

    def test_rejects_lr_mismatch(self):
        proposal = self._row("fedsa_lora", "FL+fedsa_lora")
        baseline = self._row(
            "fixed_share_b_lora", "FL+fixed_share_b_lora"
        )
        baseline["protocol_manifest"]["lr"] = 1e-3

        with self.assertRaisesRegex(ValueError, "mismatched controls"):
            paired_factor_sharing_comparisons([proposal, baseline])

    def test_cross_client_move_realization_is_not_a_cross_seed_family_key(self):
        base_metadata = {
            "partition": "dirichlet",
            "dirichlet_alpha": 0.4,
            "num_clients": 3,
            "cross_split_client_source_group_check": {
                "schema_version": 1,
                "policy": (
                    "train_then_val_then_test_global_source_group_client_owner_v1"
                ),
                "shared_source_groups": 281,
                "cross_split_client_owner_conflicts": 0,
                "alignment_moves": [{"source_group_id": "seed-specific-a"}],
                "quantity_balance_moves": [],
                "per_split": {"train": {"client_image_counts": [1, 2, 3]}},
            },
        }
        other_metadata = {
            **base_metadata,
            "cross_split_client_source_group_check": {
                **base_metadata["cross_split_client_source_group_check"],
                "alignment_moves": [{"source_group_id": "seed-specific-b"}],
                "per_split": {"train": {"client_image_counts": [2, 2, 2]}},
            },
        }

        first = _split_generation_protocol({"split_metadata": base_metadata})
        second = _split_generation_protocol({"split_metadata": other_metadata})

        self.assertEqual(_sha256_of_json(first), _sha256_of_json(second))

    def test_mia_envelope_requires_source_disjoint_sampling_audit(self):
        payload = {
            "status": "complete",
            "result_schema_version": 2,
            "mia": {
                "configuration": {"nonmember_source_policy": "unsafe_raw_test"},
                "per_client": [{"auc_roc": 0.5}],
            },
        }
        with self.assertRaisesRegex(ValueError, "source-disjoint"):
            _validate_result_envelope("unsafe.json", payload)


if __name__ == "__main__":
    unittest.main()
