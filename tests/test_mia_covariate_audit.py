"""Tests for the read-only covariate-matched MIA reanalysis."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.mia_covariate_audit import (
    ATTACK_METRICS,
    METHODS,
    SCORES,
    _aggregate,
    _cache_path,
    _expected_cache_key,
    _load_cache,
    _summary_markdown,
    _validate_pool_pair,
    build_repeated_match_plans,
    coarsened_exact_match,
    json_sha256,
    metadata_sha256,
    repeated_matched_attack,
    source_group_atomic_split,
    validate_output_location,
)


def _record(index: int, *, member: bool, loss: float | None = None) -> dict:
    # Four repeated coarsened strata exercise matching while keeping exact
    # post-match balance. Source groups repeat in pairs to exercise atomicity.
    class_name = ("airplane", "bird", "drone", "helicopter")[index % 4]
    sample_offset = 0 if member else 10_000
    source_offset = 0 if member else 20_000
    return {
        "sample_id_sha256": f"{sample_offset + index + 1:064x}",
        "image_content_sha256": f"{sample_offset + index + 30_000:064x}",
        "source_group_sha256": f"{source_offset + index // 2 + 1:064x}",
        "split": "train" if member else "test",
        "width": 1024,
        "height": 1024,
        "object_count": 1,
        "background": False,
        "class_counts": {class_name: 1},
        "classes": [class_name],
        "bbox_area_ratio_mean": 2.0 ** (-8 - (index % 2)),
        "bbox_area_ratio_min": 2.0 ** (-8 - (index % 2)),
        "bbox_area_ratio_max": 2.0 ** (-8 - (index % 2)),
        "trained_loss": float(loss if loss is not None else (0.2 if member else 0.8)),
        "initial_loss": float(0.5 + 0.001 * (index % 3)),
        "delta_loss": float(
            (loss if loss is not None else (0.2 if member else 0.8))
            - (0.5 + 0.001 * (index % 3))
        ),
    }


def _pools(count: int = 320) -> tuple[list[dict], list[dict]]:
    return (
        [_record(index, member=True) for index in range(count)],
        [_record(index, member=False) for index in range(count)],
    )


class MatchingTests(unittest.TestCase):
    def test_matching_is_deterministic_and_does_not_use_attack_scores(self):
        members, nonmembers = _pools()
        first = coarsened_exact_match(members, nonmembers, seed=123)
        changed_members = copy.deepcopy(members)
        changed_nonmembers = copy.deepcopy(nonmembers)
        for index, row in enumerate(changed_members + changed_nonmembers):
            row["trained_loss"] = 1000.0 - index
            row["initial_loss"] = -500.0 + index
            row["delta_loss"] = row["trained_loss"] - row["initial_loss"]
        second = coarsened_exact_match(changed_members, changed_nonmembers, seed=123)
        self.assertEqual(first["pairs"], second["pairs"])
        self.assertEqual(first["match_plan_sha256"], second["match_plan_sha256"])
        self.assertEqual(first["matched_pairs"], 320)
        self.assertEqual(first["post_match_balance"]["max_abs_smd"], 0.0)

    def test_source_group_atomic_split_is_disjoint_and_deterministic(self):
        members, _ = _pools()
        first = source_group_atomic_split(
            members, fraction=0.5, seed=77, label="member"
        )
        second = source_group_atomic_split(
            members, fraction=0.5, seed=77, label="member"
        )
        self.assertEqual(first, second)
        calibration, evaluation, _ = first
        self.assertGreaterEqual(min(len(calibration), len(evaluation)), 2)
        calibration_groups = {row["source_group_sha256"] for row in calibration}
        evaluation_groups = {row["source_group_sha256"] for row in evaluation}
        self.assertFalse(calibration_groups & evaluation_groups)

    def test_split_first_plans_do_not_create_cross_side_giant_components(self):
        members, nonmembers = _pools(512)
        with patch("scripts.mia_covariate_audit.ATTACK_REPEATS", 3):
            plans, balance = build_repeated_match_plans(
                members, nonmembers, client_id=0
            )
        self.assertEqual(len(plans), 3)
        self.assertEqual(len(balance), 6)
        self.assertTrue(
            all(plan["calibration_pairs"] >= 64 for plan in plans)
        )
        self.assertTrue(all(plan["evaluation_pairs"] >= 64 for plan in plans))

    def test_repeated_attack_reuses_the_same_split_for_every_score(self):
        try:
            import numpy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("NumPy/scikit-learn are unavailable in this test runtime")
        members, nonmembers = _pools()
        member_map = {row["sample_id_sha256"]: row for row in members}
        nonmember_map = {row["sample_id_sha256"]: row for row in nonmembers}
        digests = set()
        with patch("scripts.mia_covariate_audit.ATTACK_REPEATS", 3):
            plans, _ = build_repeated_match_plans(
                members, nonmembers, client_id=0
            )
            for score in SCORES:
                result = repeated_matched_attack(
                    member_map,
                    nonmember_map,
                    plans,
                    score_field=score,
                )
                digests.add(result["repeat_plan_sha256"])
                self.assertEqual(result["attack_repeats"], 3)
        self.assertEqual(len(digests), 1)


class IntegrityAndAggregationTests(unittest.TestCase):
    @staticmethod
    def _v1_report_contract() -> dict:
        method = "lora"
        control = "d" * 64
        return {
            "split_manifest_sha256": "a" * 64,
            "model_weight_sha256": "b" * 64,
            "fresh_initial_target_tensor_state_sha256": "c" * 64,
            "method_specific_fresh_control_state_sha256": {method: control},
            "methods": {
                method: {
                    "best_checkpoint_sha256": "e" * 64,
                    "fresh_control_model_state_sha256": control,
                    "clients": [{
                        "client_id": 0,
                        "reconstructed_model_state_sha256": "f" * 64,
                        "scopes": {
                            "local": {
                                "member_sample_ids_sha256": "1" * 64,
                                "nonmember_sample_ids_sha256": "2" * 64,
                                "attacks": {
                                    "trained_loss": {
                                        "member_count": 1000,
                                        "nonmember_count": 700,
                                    }
                                },
                            }
                        },
                    }],
                }
            },
        }

    def test_v1_cache_paths_and_exact_producer_keys(self):
        root = Path("/audit")
        trained_path, initial_path = _cache_path(root, "lora", 0, "member")
        self.assertEqual(
            trained_path,
            root / "loss_records/lora/client_0_member_shared_member.json",
        )
        self.assertEqual(
            initial_path,
            root / (
                "loss_records/initial_control/lora/"
                "client_0_member_shared_member.json"
            ),
        )

        report = self._v1_report_contract()
        trained_key = _expected_cache_key(
            report,
            method="lora",
            client_id=0,
            membership="member",
            initial=False,
        )
        self.assertEqual(trained_key, {
            "kind": "trained_validation_selected_personalized_model",
            "method": "lora",
            "checkpoint_sha256": "e" * 64,
            "reconstructed_client_state_sha256": "f" * 64,
            "split_manifest_sha256": "a" * 64,
            "client_id": 0,
            "scope": "member_shared",
            "membership": "member",
            "sample_ids_sha256": "1" * 64,
            "sample_count": 1000,
            "img_size": 640,
            "batch_size": 1,
        })
        initial_key = _expected_cache_key(
            report,
            method="lora",
            client_id=0,
            membership="nonmember",
            initial=True,
        )
        self.assertEqual(initial_key, {
            "kind": "method_specific_fresh_target_initialization_control",
            "control_method": "lora",
            "initial_target_tensor_state_sha256": "c" * 64,
            "control_model_state_sha256": "d" * 64,
            "model_weight_sha256": "b" * 64,
            "split_manifest_sha256": "a" * 64,
            "client_id": 0,
            "scope": "local",
            "membership": "nonmember",
            "sample_ids_sha256": "2" * 64,
            "sample_count": 700,
            "img_size": 640,
            "batch_size": 1,
        })

    def test_member_nonmember_source_or_content_overlap_is_rejected(self):
        members, nonmembers = _pools(160)
        nonmembers[0]["source_group_sha256"] = members[0]["source_group_sha256"]
        with self.assertRaisesRegex(ValueError, "source overlap"):
            _validate_pool_pair(members, nonmembers)

        members, nonmembers = _pools(160)
        nonmembers[0]["image_content_sha256"] = members[0]["image_content_sha256"]
        with self.assertRaisesRegex(ValueError, "content overlap"):
            _validate_pool_pair(members, nonmembers)

    def test_cache_envelope_detects_record_tampering(self):
        trained = _record(0, member=True)
        cache_record = {
            key: value for key, value in trained.items()
            if key not in SCORES
        }
        cache_record["loss"] = trained["trained_loss"]
        cache_record["loss_components"] = {}
        records = [cache_record]
        expected_key = {
            "membership": "member",
            "sample_count": 1,
            "sample_ids_sha256": json_sha256([cache_record["sample_id_sha256"]]),
        }
        payload = {
            "schema_version": 1,
            "cache_key": expected_key,
            "records_sha256": json_sha256(records),
            "records": records,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(_load_cache(path, expected_key)), 1)

            # v1 stores raw-COCO bbox area divided by image area without
            # clipping the bbox first, so its valid producer contract permits
            # a ratio above one.
            for key in (
                "bbox_area_ratio_min",
                "bbox_area_ratio_mean",
                "bbox_area_ratio_max",
            ):
                payload["records"][0][key] = 1.25
            payload["records_sha256"] = json_sha256(payload["records"])
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(_load_cache(path, expected_key)), 1)

            payload["records"][0]["loss"] = 99.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid, stale"):
                _load_cache(path, expected_key)

    def test_cache_key_provenance_mismatch_is_rejected_even_if_self_hash_is_valid(self):
        trained = _record(0, member=True)
        record = {key: value for key, value in trained.items() if key not in SCORES}
        record.update(loss=trained["trained_loss"], loss_components={})
        expected_key = {
            "kind": "trained",
            "method": "lora",
            "membership": "member",
            "sample_count": 1,
            "sample_ids_sha256": json_sha256([record["sample_id_sha256"]]),
        }
        wrong_key = {**expected_key, "method": "fedsa_lora"}
        payload = {
            "schema_version": 1,
            "cache_key": wrong_key,
            "records_sha256": json_sha256([record]),
            "records": [record],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance-mismatched"):
                _load_cache(path, expected_key)

    def test_method_metadata_fingerprint_excludes_only_attack_scores(self):
        members, _ = _pools(4)
        changed_score = copy.deepcopy(members)
        changed_score[0]["trained_loss"] += 99.0
        self.assertEqual(metadata_sha256(members), metadata_sha256(changed_score))
        changed_metadata = copy.deepcopy(members)
        changed_metadata[0]["width"] = 999
        self.assertNotEqual(metadata_sha256(members), metadata_sha256(changed_metadata))

    def test_output_location_is_exact_and_completed_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            expected = root / "security_audit_v2_covariate_multiseed"
            self.assertEqual(validate_output_location(root, expected), expected.resolve())
            with self.assertRaisesRegex(ValueError, "writes only"):
                validate_output_location(root, root / "other")
            expected.mkdir()
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                validate_output_location(root, expected)

    def test_aggregate_uses_exactly_three_seed_replicates(self):
        reports = {}
        for seed, base in ((42, 0.50), (43, 0.60), (44, 0.70)):
            methods = {}
            for method_index, method in enumerate(METHODS):
                macro = {}
                for score in SCORES:
                    macro[score] = {}
                    for metric in ATTACK_METRICS:
                        macro[score][metric] = {
                            "client_macro_mean": base + 0.01 * method_index,
                            "client_sample_sd": 0.02,
                        }
                methods[method] = {"macro": macro}
            reports[seed] = {"methods": methods}
        rows, paired = _aggregate(reports)
        target = next(
            row for row in rows
            if row["method"] == "full_ft"
            and row["score"] == "trained_loss"
            and row["metric"] == "auc_roc"
        )
        self.assertEqual(target["n_paired_replicates"], 3)
        self.assertAlmostEqual(target["mean"], 0.60)
        self.assertAlmostEqual(target["replicate_sample_sd"], 0.10)
        self.assertTrue(paired)

    def test_summary_labels_calibration_target_and_achieved_evaluation_fpr(self):
        rows = []
        for method in METHODS:
            for score in SCORES:
                for metric in ATTACK_METRICS:
                    rows.append({
                        "method": method,
                        "score": score,
                        "metric": metric,
                        "mean": 0.5,
                        "replicate_sample_sd": 0.01,
                    })
        balance = []
        for seed in (42, 43, 44):
            for client_id in range(3):
                for partition in ("calibration", "evaluation"):
                    balance.append({
                        "seed": seed,
                        "client_id": client_id,
                        "partition": partition,
                        "matched_pairs": 100,
                        "retention": 0.5,
                        "post_max_abs_smd": 0.0,
                        "nonmember_fpr_resolution": 0.01,
                    })
        rendered = _summary_markdown(rows, balance)
        self.assertIn("Eval TPR/FPR at cal-target 5%", rendered)
        self.assertIn("Eval TPR/FPR at cal-target 10%", rendered)
        self.assertIn("exploratory post-hoc", rendered)


if __name__ == "__main__":
    unittest.main()
