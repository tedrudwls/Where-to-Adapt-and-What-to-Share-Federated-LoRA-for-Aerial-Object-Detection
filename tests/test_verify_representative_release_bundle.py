"""Tests for safe representative release-archive verification."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_representative_release_bundle as builder
from scripts import verify_representative_release_bundle as target


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


class RepresentativeReleaseVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / target.ROOT
        (self.stage / "checkpoint").mkdir(parents=True)
        (self.stage / "metadata").mkdir()
        self.checkpoint = b"synthetic-public-checkpoint"
        identity_patch = mock.patch.multiple(
            target,
            PUBLIC_CHECKPOINT_SHA256=_sha(self.checkpoint),
            PUBLIC_CHECKPOINT_BYTES=len(self.checkpoint),
            PUBLIC_TENSOR_FINGERPRINT_SHA256="c" * 64,
            PUBLIC_TENSOR_COUNT=3,
        )
        identity_patch.start()
        self.addCleanup(identity_patch.stop)
        (self.stage / "checkpoint" / target.CHECKPOINT_NAME).write_bytes(
            self.checkpoint
        )
        replay_source = builder.COMMITTED_REPLAY
        self.replay = replay_source.read_bytes()
        self.assertEqual(len(self.replay), target.REPLAY_BYTES)
        self.assertEqual(_sha(self.replay), target.REPLAY_SHA256)
        (self.stage / "metadata" / target.REPLAY_NAME).write_bytes(self.replay)
        (self.stage / "README.md").write_text("public readme\n", encoding="utf-8")
        (self.stage / "LICENSE").write_text("license\n", encoding="utf-8")
        (self.stage / "THIRD_PARTY_NOTICES.md").write_text(
            "third party\n", encoding="utf-8"
        )
        described = [
            (
                f"checkpoint/{target.CHECKPOINT_NAME}",
                "sanitized_validation_selected_checkpoint",
            ),
            (
                f"metadata/{target.REPLAY_NAME}",
                "path_free_test_replay_manifest",
            ),
            ("README.md", "bundle_readme"),
            ("LICENSE", "license"),
            ("THIRD_PARTY_NOTICES.md", "third_party_notices"),
        ]
        files = []
        for name, role in described:
            raw = (self.stage / name).read_bytes()
            files.append({
                "path": name, "role": role, "bytes": len(raw), "sha256": _sha(raw)
            })
        files[0]["historical_source"] = {
            "bytes": target.HISTORICAL_CHECKPOINT_BYTES,
            "sha256": target.HISTORICAL_CHECKPOINT_SHA256,
        }
        files[0]["path_replacements"] = [
            {
                "json_pointer": pointer,
                "historical_value_sha256": "b" * 64,
                "public_value": "external/rtdetr-l.pt",
            }
            for pointer in (
                "/architecture/model_weight_path", "/experiment/model_weights"
            )
        ]
        files[0]["tensor_fingerprint"] = {
            "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
            "tensor_count": 3,
            "sha256": "c" * 64,
        }
        manifest = {
            "schema_version": 1,
            "bundle_id": "seed42-fedlora-a-r8-a04-replay",
            "release_version": "1.0.0",
            "experiment_id": "seed_42/fl_fedsa_lora_r8_a0.4",
            "method": "FedLoRA-A (Share-A / local B)",
            "source_repository": {
                "url": target.SOURCE_REPOSITORY,
                "commit": "a" * 40,
            },
            "files": files,
            "historical_split_manifest_sha256": target.HISTORICAL_SPLIT_SHA256,
            "external_inputs": {
                "pretrained_model": {
                    "included": False,
                    "file_name": "rtdetr-l.pt",
                    "sha256": target.PRETRAINED_SHA256,
                },
                "dataset": {
                    "included": False,
                    "doi": "10.17632/cd5z895tr2.1",
                    "test_images": 2241,
                    "test_image_tree_sha256": target.TEST_IMAGE_TREE_SHA256,
                },
            },
            "expected_metrics": {
                "metric_scale": "0_to_1",
                "client_local_macro": {
                    "AP": 0.6190019159385891,
                    "AP50": 0.902681661287866,
                    "AP75": 0.6780732838875302,
                },
                "common_macro": {
                    "AP": 0.5620845765652397,
                    "AP50": 0.8576226357547724,
                    "AP75": 0.6074470685556409,
                },
                "absolute_tolerance": 1e-6,
            },
        }
        (self.stage / "BUNDLE_MANIFEST.json").write_bytes(_json(manifest))
        checksum_names = sorted(
            name for name, _ in described
        ) + ["BUNDLE_MANIFEST.json"]
        checksum_names.sort()
        (self.stage / "SHA256SUMS").write_text(
            "".join(
                f"{_sha((self.stage / name).read_bytes())}  {name}\n"
                for name in checksum_names
            ),
            encoding="ascii",
        )
        self.archive = self.root / target.ARCHIVE_NAME
        builder._archive_tree(self.stage, self.archive)
        self.outer = self.root / f"{target.ARCHIVE_NAME}.sha256"
        self.outer.write_text(
            f"{target.sha256_file(self.archive)}  {target.ARCHIVE_NAME}\n",
            encoding="ascii",
        )

    def test_valid_bundle_passes_without_loading_checkpoint(self):
        report = target.verify(self.archive, self.outer)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            report["public_checkpoint"]["sha256"], _sha(self.checkpoint)
        )
        self.assertEqual(report["source_commit"], "a" * 40)

    def test_outer_checksum_mismatch_stops_before_tar_read(self):
        self.outer.write_text(
            f"{'0' * 64}  {target.ARCHIVE_NAME}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(target.VerificationError, "Outer archive SHA"):
            target.verify(self.archive, self.outer)

    def test_path_traversal_name_is_rejected(self):
        with self.assertRaisesRegex(target.VerificationError, "Unsafe tar member"):
            target._safe_name("../escape")

    def test_outer_archive_symlink_is_rejected(self):
        link = self.root / "linked-archive.tar.gz"
        link.symlink_to(self.archive)
        with self.assertRaisesRegex(target.VerificationError, "must not be symlinks"):
            target.verify(link, self.outer)

    def test_extra_member_is_rejected_before_payload_verification(self):
        (self.stage / "unexpected.bin").write_bytes(b"extra")
        self.archive.unlink()
        builder._archive_tree(self.stage, self.archive)
        self.outer.write_text(
            f"{target.sha256_file(self.archive)}  {target.ARCHIVE_NAME}\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(target.VerificationError, "member count"):
            target.verify(self.archive, self.outer)

    def test_self_consistent_checkpoint_rewrite_still_fails_public_identity_pin(self):
        checkpoint_path = self.stage / "checkpoint" / target.CHECKPOINT_NAME
        checkpoint_path.write_bytes(b"different-public-checkpoint")
        manifest_path = self.stage / "BUNDLE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next(
            row for row in manifest["files"]
            if row["path"] == f"checkpoint/{target.CHECKPOINT_NAME}"
        )
        record["bytes"] = checkpoint_path.stat().st_size
        record["sha256"] = _sha(checkpoint_path.read_bytes())
        manifest_path.write_bytes(_json(manifest))
        checksum_names = sorted(
            name for name in (
                "BUNDLE_MANIFEST.json", "README.md", "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                f"checkpoint/{target.CHECKPOINT_NAME}",
                f"metadata/{target.REPLAY_NAME}",
            )
        )
        (self.stage / "SHA256SUMS").write_text(
            "".join(
                f"{_sha((self.stage / name).read_bytes())}  {name}\n"
                for name in checksum_names
            ),
            encoding="ascii",
        )
        self.archive.unlink()
        builder._archive_tree(self.stage, self.archive)
        self.outer.write_text(
            f"{target.sha256_file(self.archive)}  {target.ARCHIVE_NAME}\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            target.VerificationError, "Public checkpoint identity changed"
        ):
            target.verify(self.archive, self.outer)


if __name__ == "__main__":
    unittest.main()
