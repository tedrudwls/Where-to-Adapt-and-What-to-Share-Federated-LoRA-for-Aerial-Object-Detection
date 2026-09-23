"""Static integrity checks for the 12-checkpoint paper-core allowlist."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json"
INDEX_PATH = PROJECT_ROOT / "artifacts" / "checkpoint_index.json"


class PaperCoreReleaseSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    def test_spec_selects_exact_primary_method_seed_matrix(self):
        records = self.spec["records"]
        expected = {
            (seed, method)
            for seed in (42, 43, 44)
            for method in (
                "full_ft",
                "lora",
                "fedsa_lora",
                "fixed_share_b_lora",
            )
        }
        observed = {
            (record["training_seed"], record["internal_method"])
            for record in records
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len(records), 12)
        self.assertEqual(len({record["experiment_id"] for record in records}), 12)
        self.assertTrue(
            all(
                record["training_seed"] == record["partition_seed"]
                for record in records
            )
        )
        expected_names = {
            "full_ft": "FL Full FT",
            "lora": "FedLoRA-AB",
            "fedsa_lora": "FedLoRA-A",
            "fixed_share_b_lora": "FedLoRA-B",
        }
        self.assertTrue(
            all(
                record["paper_method"]
                == expected_names[record["internal_method"]]
                for record in records
            )
        )

    def test_spec_exactly_matches_the_immutable_checkpoint_index(self):
        actual_index_sha = hashlib.sha256(INDEX_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            self.spec["checkpoint_index"]["sha256"], actual_index_sha
        )
        indexed = {
            record["experiment_id"]: record for record in self.index["records"]
        }
        total_bytes = 0
        for record in self.spec["records"]:
            source = indexed[record["experiment_id"]]
            historical = record["historical"]
            self.assertEqual(record["internal_method"], source["method"])
            self.assertEqual(record["training_seed"], source["training_seed"])
            self.assertEqual(record["partition_seed"], source["partition_seed"])
            self.assertEqual(record["rank"], source["rank"])
            self.assertEqual(record["selected_round"], source["selected_at"])
            self.assertEqual(source["selected_unit"], "round")
            self.assertEqual(
                record["split_manifest_sha256"],
                source["split_manifest_sha256"],
            )
            self.assertEqual(
                historical["project_relative_path"],
                source["historical_project_relative_path"],
            )
            self.assertEqual(historical["bytes"], source["bytes"])
            self.assertEqual(historical["sha256"], source["sha256"])
            self.assertEqual(record["public_file_name"], source["release_asset_name"])
            self.assertIsNone(record["public_identity"])
            total_bytes += historical["bytes"]
        self.assertEqual(total_bytes, 423_860_180)
        self.assertEqual(total_bytes, self.spec["historical_total_bytes"])

    def test_spec_contains_no_absolute_or_private_paths(self):
        raw = SPEC_PATH.read_text(encoding="utf-8")
        for marker in ("/home/", "/Users/", "gpuadmin", "file://"):
            self.assertNotIn(marker, raw)
        for record in self.spec["records"]:
            relative = Path(record["historical"]["project_relative_path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)


if __name__ == "__main__":
    unittest.main()
