"""Static integrity checks for the 12-checkpoint paper-core allowlist."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json"
INDEX_PATH = PROJECT_ROOT / "artifacts" / "checkpoint_index.json"
AUDIT_PATH = PROJECT_ROOT / "artifacts" / "paper_core_input_audit.json"


class PaperCoreReleaseSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_spec_selects_exact_primary_method_seed_matrix(self):
        self.assertEqual(self.spec["status"], "public_identities_pinned")
        self.assertEqual(
            self.spec["archive_name"],
            "fedlora-paper-core-checkpoints-v1.0.0.tar.gz",
        )
        self.assertEqual(self.spec["archive_root"], "paper-core-checkpoints")
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
            public = record["public_identity"]
            self.assertEqual(
                set(public),
                {
                    "bytes",
                    "sha256",
                    "tensor_fingerprint",
                    "path_replacements",
                    "residual_absolute_path_count",
                },
            )
            self.assertGreater(public["bytes"], 0)
            self.assertEqual(len(public["sha256"]), 64)
            self.assertEqual(public["residual_absolute_path_count"], 0)
            self.assertEqual(
                {row["json_pointer"] for row in public["path_replacements"]},
                {
                    "/architecture/model_weight_path",
                    "/experiment/model_weights",
                },
            )
            total_bytes += historical["bytes"]
        self.assertEqual(total_bytes, 423_860_180)
        self.assertEqual(total_bytes, self.spec["historical_total_bytes"])
        public_total = sum(
            record["public_identity"]["bytes"] for record in self.spec["records"]
        )
        self.assertEqual(public_total, 423_859_412)
        self.assertEqual(public_total, self.spec["public_total_bytes"])

    def test_identity_audit_receipt_is_path_free_and_pinned(self):
        receipt = self.spec["identity_audit"]
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["report_bytes"], 37_737)
        self.assertEqual(
            receipt["report_sha256"],
            "c41bd0091da3ee727477df5c96377147f04316340794fef5d0c681cbccf80be5",
        )
        self.assertEqual(receipt["protected_file_count"], 14)
        self.assertEqual(
            self.spec["serialization_runtime"],
            {"python": "3.9.18", "torch": "2.5.1+cu124"},
        )
        self.assertEqual(receipt["report_bytes"], AUDIT_PATH.stat().st_size)
        self.assertEqual(
            receipt["report_sha256"],
            hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest(),
        )
        portable_sidecar = (
            PROJECT_ROOT / "artifacts" / "paper_core_input_audit.json.sha256"
        ).read_text(encoding="ascii")
        self.assertEqual(
            portable_sidecar,
            f"{receipt['report_sha256']}  paper_core_input_audit.json\n",
        )

    def test_audit_candidates_match_pinned_spec_record_by_record(self):
        audit_records = self.audit["records"]
        spec_records = self.spec["records"]
        self.assertEqual(
            [row["experiment_id"] for row in audit_records],
            [row["experiment_id"] for row in spec_records],
        )
        self.assertEqual(len(audit_records), 12)
        for discovered, pinned in zip(audit_records, spec_records):
            candidate = discovered["public_checkpoint_candidate"]
            self.assertEqual(discovered["method"], pinned["internal_method"])
            self.assertEqual(discovered["training_seed"], pinned["training_seed"])
            self.assertEqual(discovered["partition_seed"], pinned["partition_seed"])
            self.assertEqual(discovered["rank"], pinned["rank"])
            self.assertEqual(candidate["file_name"], pinned["public_file_name"])
            self.assertEqual(
                candidate["release_asset_name"], pinned["public_file_name"]
            )
            self.assertEqual(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"file_name", "release_asset_name"}
                },
                pinned["public_identity"],
            )

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
