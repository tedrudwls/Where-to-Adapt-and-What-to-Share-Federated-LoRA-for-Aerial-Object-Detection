"""Tests for the read-only 12-checkpoint paper-core input audit."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_paper_core_release_inputs as target
from scripts.verify_checkpoint_assets import load_index


class FakeTensor:
    def __init__(self, elements: int = 2, element_bytes: int = 4):
        self.elements = elements
        self.element_bytes = element_bytes

    def numel(self):
        return self.elements

    def element_size(self):
        return self.element_bytes


class FakeTorch:
    __version__ = "9.9.9+test"

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(seed: int, method: str, path: str, raw: bytes) -> dict:
    contract = target.METHOD_CONTRACTS[method]
    experiment_suffix = contract["experiment_suffix"]
    return {
        "experiment_id": target._expected_experiment_id(seed, method),
        "mode": "fl",
        "method": method,
        "partition": "dirichlet",
        "partition_seed": seed,
        "training_seed": seed,
        "rank": contract["rank"],
        "client_id": None,
        "selected_unit": "round",
        "selected_at": 19 if method == "full_ft" else 20,
        "historical_project_relative_path": path,
        "release_asset_name": (
            f"seed_{seed}__{experiment_suffix}__best_federated.pt"
        ),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "pretrained_sha256": target.PRETRAINED_SHA256,
        "split_manifest_sha256": hashlib.sha256(f"split-{seed}".encode()).hexdigest(),
    }


def _payload(record: dict) -> dict:
    method = record["method"]
    contract = target.METHOD_CONTRACTS[method]
    if method == "full_ft":
        shared = {"model.weight": FakeTensor()}
        local = None
    elif method == "lora":
        shared = {
            "model.block.lora_A": FakeTensor(),
            "model.block.lora_B": FakeTensor(),
            "model.score_head.weight": FakeTensor(),
        }
        local = None
    elif method == "fedsa_lora":
        shared = {
            "model.block.lora_A": FakeTensor(),
            "model.score_head.weight": FakeTensor(),
        }
        local = [{"model.block.lora_B": FakeTensor()} for _ in range(3)]
    else:
        shared = {
            "model.block.lora_B": FakeTensor(),
            "model.class_embed.weight": FakeTensor(),
        }
        local = [{"model.block.lora_A": FakeTensor()} for _ in range(3)]
    compatibility = target._expected_compatibility(
        record["training_seed"], method
    )
    return {
        "schema_version": 5,
        "checkpoint_kind": "federated_personalized",
        "resume_capability": "evaluation_only_no_optimizer_scheduler_or_rng_state",
        "selection": "best_macro_client_local_val_AP",
        "round": record["selected_at"],
        "best_round": record["selected_at"],
        "training_rounds_executed": 20,
        "fl_method": method,
        "num_clients": 3,
        "class_names": list(target.CLASS_NAMES),
        "split_manifest_sha256": record["split_manifest_sha256"],
        "nonfloating_state_policy": "retain_previous_server_value",
        "federated_payload_policy": contract["payload_policy"],
        "shared_lora_factor_role": contract["shared_role"],
        "local_lora_factor_role": contract["local_role"],
        "shared_state": shared,
        "local_personalized_states": local,
        "compatibility": compatibility,
    }


class CoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _fixture(self):
        project = self.root / "project"
        project.mkdir()
        records = []
        payloads = {}
        for seed in target.SEEDS:
            for method in target.METHOD_ORDER:
                relative = f"results/seed_{seed}/{method}/best_federated.pt"
                raw = f"checkpoint-{seed}-{method}".encode()
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                record = _record(seed, method, relative, raw)
                records.append(record)
                payloads[str(path.resolve())] = _payload(record)
        index = project / "artifacts" / "checkpoint_index.json"
        index.parent.mkdir()
        index.write_text(
            json.dumps({
                "schema_version": 1,
                "record_count": 12,
                "total_bytes": sum(row["bytes"] for row in records),
                "records": records,
            }, sort_keys=True),
            encoding="utf-8",
        )
        spec_records = []
        for record in records:
            spec_records.append({
                "experiment_id": record["experiment_id"],
                "paper_method": target.PAPER_METHOD_NAMES[record["method"]],
                "internal_method": record["method"],
                "training_seed": record["training_seed"],
                "partition_seed": record["partition_seed"],
                "rank": record["rank"],
                "selected_round": record["selected_at"],
                "split_manifest_sha256": record["split_manifest_sha256"],
                "historical": {
                    "project_relative_path": record["historical_project_relative_path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                },
                "public_file_name": record["release_asset_name"],
                "public_identity": None,
            })
        spec = project / "artifacts" / "paper_core_checkpoint_release_spec.json"
        spec.write_text(
            json.dumps({
                "schema_version": 1,
                "release_id": "paper-core-checkpoints-v1.0.0",
                "status": "public_identity_discovery_required",
                "checkpoint_index": {
                    "path": "artifacts/checkpoint_index.json",
                    "sha256": _sha256(index),
                },
                "external_pretrained_model": {
                    "included": False,
                    "file_name": "rtdetr-l.pt",
                    "sha256": target.PRETRAINED_SHA256,
                },
                "expected_record_count": 12,
                "historical_total_bytes": sum(row["bytes"] for row in records),
                "records": spec_records,
            }, sort_keys=True),
            encoding="utf-8",
        )
        return project, index, spec, records, payloads

    def test_committed_index_and_release_spec_define_exact_grid(self):
        index_path = target.PROJECT_ROOT / "artifacts" / "checkpoint_index.json"
        spec_path = target.DEFAULT_SPEC
        records = target.select_core_records(load_index(index_path))
        self.assertEqual(len(records), 12)
        self.assertEqual(
            [(row["training_seed"], row["method"]) for row in records],
            [
                (seed, method)
                for seed in target.SEEDS
                for method in target.METHOD_ORDER
            ],
        )
        index_identity = {
            "relative_path": "artifacts/checkpoint_index.json",
            "sha256": _sha256(index_path),
        }
        target._validate_spec_against_index(
            target._load_release_spec(spec_path), records, index_identity
        )

    def test_code_checkout_metadata_is_named_relative_to_code_not_source_project(self):
        unrelated_source_project = self.root / "historical-project"
        unrelated_source_project.mkdir()
        self.assertEqual(
            target._metadata_relative_path(
                target.DEFAULT_SPEC.resolve(),
                unrelated_source_project.resolve(),
                "fallback.json",
            ),
            "artifacts/paper_core_checkpoint_release_spec.json",
        )

    def test_method_contracts_distinguish_all_four_state_layouts(self):
        for method in target.METHOD_ORDER:
            with self.subTest(method=method):
                record = _record(42, method, f"x/{method}.pt", b"checkpoint")
                summary = target.validate_checkpoint_contract(
                    _payload(record), record, FakeTorch
                )
                self.assertEqual(
                    summary["shared_lora_factor_role"],
                    target.METHOD_CONTRACTS[method]["shared_role"],
                )
                self.assertEqual(
                    len(summary["local_personalized_states"]),
                    3 if method in ("fedsa_lora", "fixed_share_b_lora") else 0,
                )

    def test_factor_contract_rejects_share_a_with_a_local_state(self):
        record = _record(42, "fedsa_lora", "x/a.pt", b"checkpoint")
        payload = _payload(record)
        payload["local_personalized_states"][0] = {
            "model.block.lora_A": FakeTensor()
        }
        with self.assertRaisesRegex(target.CoreAuditError, "unexpected LoRA-A"):
            target.validate_checkpoint_contract(payload, record, FakeTorch)

    def test_end_to_end_writes_path_free_report_and_preserves_sources(self):
        project, index, spec, records, payloads = self._fixture()
        output = self.root / "paper_core_input_audit.json"
        public_identity = {
            "bytes": target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_BYTES,
            "sha256": target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_SHA256,
            "tensor_fingerprint": {
                "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
                "tensor_count": (
                    target.release_builder.evaluation.FROZEN_PUBLIC_TENSOR_COUNT
                ),
                "sha256": (
                    target.release_builder.evaluation
                    .FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256
                ),
            },
            "path_replacements": [
                {
                    "json_pointer": "/architecture/model_weight_path",
                    "historical_value_sha256": "c" * 64,
                    "public_value": "external/rtdetr-l.pt",
                },
                {
                    "json_pointer": "/experiment/model_weights",
                    "historical_value_sha256": "d" * 64,
                    "public_value": "external/rtdetr-l.pt",
                },
            ],
            "residual_absolute_path_count": 0,
        }
        with mock.patch.object(target, "_load_torch", return_value=FakeTorch), mock.patch.object(
            target,
            "_restricted_load",
            side_effect=lambda source, unused: copy.deepcopy(payloads[str(source)]),
        ), mock.patch.object(
            target,
            "_discover_public_identity",
            return_value=public_identity,
        ):
            report = target.run_audit(project, index, spec, output)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["records"]), 12)
        self.assertEqual(report["protected_source_gate"]["protected_file_count"], 14)
        self.assertEqual(
            report["serialization_runtime"],
            {"python": mock.ANY, "torch": "9.9.9+test"},
        )
        self.assertEqual(
            report["records"][0]["public_checkpoint_candidate"]["file_name"],
            records[0]["release_asset_name"],
        )
        self.assertTrue(output.is_file())
        self.assertNotIn(str(self.root), output.read_text(encoding="utf-8"))
        for record in records:
            path = project / record["historical_project_relative_path"]
            self.assertEqual(_sha256(path), record["sha256"])

    def test_path_free_gate_allows_only_the_two_reviewed_json_pointers(self):
        target._assert_path_free_report({
            "path_replacements": [
                {"json_pointer": "/architecture/model_weight_path"},
                {"json_pointer": "/experiment/model_weights"},
            ]
        })
        with self.assertRaisesRegex(
            target.CoreAuditError, "absolute local filesystem path"
        ):
            target._assert_path_free_report({
                "path_replacements": [
                    {"json_pointer": "/unexpected/absolute-looking/value"}
                ]
            })
        with self.assertRaisesRegex(
            target.CoreAuditError, "absolute local filesystem path"
        ):
            target._assert_path_free_report({"value": "/home/user/private.pt"})

    def test_existing_representative_public_identity_is_a_cross_check(self):
        record = {
            "experiment_id": target.release_builder.evaluation.TARGET_EXPERIMENT_ID
        }
        valid = {
            "bytes": target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_BYTES,
            "sha256": target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_SHA256,
            "tensor_fingerprint": {
                "tensor_count": (
                    target.release_builder.evaluation.FROZEN_PUBLIC_TENSOR_COUNT
                ),
                "sha256": (
                    target.release_builder.evaluation
                    .FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256
                ),
            },
        }
        target._assert_existing_representative_identity(record, valid)
        invalid = copy.deepcopy(valid)
        invalid["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            target.CoreAuditError, "does not reproduce the published"
        ):
            target._assert_existing_representative_identity(record, invalid)

    def test_changed_source_fails_before_report_is_written(self):
        project, index, spec, records, payloads = self._fixture()
        output = self.root / "paper_core_input_audit.json"
        first_checkpoint = project / records[0]["historical_project_relative_path"]
        calls = 0

        def mutate_once(payload, unused):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_checkpoint.write_bytes(b"mutated")
            return {
                "bytes": (
                    target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_BYTES
                ),
                "sha256": (
                    target.release_builder.evaluation.FROZEN_PUBLIC_CHECKPOINT_SHA256
                ),
                "tensor_fingerprint": {
                    "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
                    "tensor_count": (
                        target.release_builder.evaluation.FROZEN_PUBLIC_TENSOR_COUNT
                    ),
                    "sha256": (
                        target.release_builder.evaluation
                        .FROZEN_PUBLIC_TENSOR_FINGERPRINT_SHA256
                    ),
                },
                "path_replacements": [],
                "residual_absolute_path_count": 0,
            }

        with mock.patch.object(target, "_load_torch", return_value=FakeTorch), mock.patch.object(
            target,
            "_restricted_load",
            side_effect=lambda source, unused: copy.deepcopy(payloads[str(source)]),
        ), mock.patch.object(target, "_discover_public_identity", side_effect=mutate_once):
            with self.assertRaisesRegex(target.CoreAuditError, "changed during audit"):
                target.run_audit(project, index, spec, output)
        self.assertFalse(output.exists())

    def test_release_spec_rejects_a_prepinned_public_identity(self):
        project, index, spec, records, payloads = self._fixture()
        payload = json.loads(spec.read_text(encoding="utf-8"))
        payload["records"][0]["public_identity"] = {"sha256": "a" * 64}
        spec.write_text(json.dumps(payload), encoding="utf-8")
        selected = target.select_core_records(load_index(index))
        index_identity = {
            "relative_path": "artifacts/checkpoint_index.json",
            "sha256": _sha256(index),
        }
        with self.assertRaisesRegex(target.CoreAuditError, "Release-spec/index mismatch"):
            target._validate_spec_against_index(
                target._load_release_spec(spec), selected, index_identity
            )


if __name__ == "__main__":
    unittest.main()
