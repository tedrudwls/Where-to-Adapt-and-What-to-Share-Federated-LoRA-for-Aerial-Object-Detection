"""Unit tests for the representative public release builder."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import build_representative_release_bundle as target
from tests.test_evaluate_checkpoint import RepresentativeFixture


class RepresentativeReleaseBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = RepresentativeFixture(Path(self.temporary.name))

    def test_replay_manifest_is_compact_path_free_and_deterministic(self):
        first = target.build_replay_manifest(self.fixture.manifest_payload)
        second = target.build_replay_manifest(
            copy.deepcopy(self.fixture.manifest_payload)
        )
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(
            first["historical_split_manifest"]["sha256"],
            target.evaluation.FROZEN_SPLIT_SHA256,
        )
        self.assertEqual(
            [len(row["image_ids"]) for row in first["test"]["client_image_ids"]],
            [1, 1, 1],
        )
        self.assertNotIn("data_root", str(first))
        absolute_values = [
            value
            for _, value in target._walk_strings(first)
            if target._path_string(value)
        ]
        self.assertEqual(absolute_values, [])

    def test_checkpoint_sanitizer_changes_only_two_reviewed_paths(self):
        payload = {
            "schema_version": 5,
            "architecture": {"model_weight_path": "/srv/model/rtdetr-l.pt"},
            "experiment": {"model_weights": "/srv/model/rtdetr-l.pt"},
            "nested": {"ordinary": "not/a/rooted/path", "value": 7},
        }
        public, replacements = target.sanitize_checkpoint_payload(payload)
        self.assertEqual(payload["architecture"]["model_weight_path"], "/srv/model/rtdetr-l.pt")
        self.assertEqual(
            public["architecture"]["model_weight_path"],
            target.PUBLIC_PRETRAINED_PATH,
        )
        self.assertEqual(
            public["experiment"]["model_weights"],
            target.PUBLIC_PRETRAINED_PATH,
        )
        self.assertEqual(public["nested"], payload["nested"])
        self.assertEqual(
            {row["json_pointer"] for row in replacements},
            {"/architecture/model_weight_path", "/experiment/model_weights"},
        )

    def test_checkpoint_sanitizer_rejects_an_unreviewed_absolute_path(self):
        payload = {
            "architecture": {"model_weight_path": "/srv/model/rtdetr-l.pt"},
            "experiment": {"model_weights": "/srv/model/rtdetr-l.pt"},
            "unexpected": "/home/user/secret.txt",
        }
        with self.assertRaisesRegex(target.BundleError, "absolute-path inventory"):
            target.sanitize_checkpoint_payload(payload)

    def test_checkpoint_sanitizer_rejects_an_absolute_mapping_key(self):
        payload = {
            "architecture": {"model_weight_path": "/srv/model/rtdetr-l.pt"},
            "experiment": {"model_weights": "/srv/model/rtdetr-l.pt"},
            "/home/user/private-key": "value",
        }
        with self.assertRaisesRegex(target.BundleError, "absolute-path inventory"):
            target.sanitize_checkpoint_payload(payload)

    def test_checkpoint_sanitizer_rejects_an_unreviewed_windows_path(self):
        payload = {
            "architecture": {"model_weight_path": r"C:\\models\\rtdetr-l.pt"},
            "experiment": {"model_weights": r"C:\\models\\rtdetr-l.pt"},
            "unexpected": r"D:\\private\\secret.txt",
        }
        with self.assertRaisesRegex(target.BundleError, "absolute-path inventory"):
            target.sanitize_checkpoint_payload(payload)

    def test_checkpoint_sanitizer_rejects_an_unc_path(self):
        payload = {
            "architecture": {"model_weight_path": "/srv/model/rtdetr-l.pt"},
            "experiment": {"model_weights": "/srv/model/rtdetr-l.pt"},
            "unexpected": r"\\server\share\secret.txt",
        }
        with self.assertRaisesRegex(target.BundleError, "absolute-path inventory"):
            target.sanitize_checkpoint_payload(payload)

    def test_binary_marker_gate_avoids_short_windows_pattern_false_positive(self):
        target._assert_no_private_checkpoint_markers(
            b"tensor-bytes-\x00C:/\xff-random", ["/srv/model/rtdetr-l.pt"]
        )

    def test_binary_marker_gate_rejects_a_reviewed_historical_path(self):
        with self.assertRaisesRegex(target.BundleError, "private path marker"):
            target._assert_no_private_checkpoint_markers(
                b"prefix-/srv/model/rtdetr-l.pt-suffix",
                ["/srv/model/rtdetr-l.pt"],
            )

    def test_bundle_notice_uses_an_immutable_repository_link(self):
        raw = target._bundle_third_party_notice(
            target.PROJECT_ROOT, "a" * 40
        )
        text = raw.decode("utf-8")
        self.assertNotIn("](docs/CHECKPOINTS.md)", text)
        self.assertIn(f"/blob/{'a' * 40}/docs/CHECKPOINTS.md", text)

    def test_replay_manifest_rejects_duplicate_test_assignment(self):
        payload = copy.deepcopy(self.fixture.manifest_payload)
        payload["clients"][1]["splits"]["test"]["image_ids"] = [1]
        with self.assertRaisesRegex(target.BundleError, "disjoint inventory cover"):
            target.build_replay_manifest(payload)

    def test_verified_reader_rejects_a_symlink(self):
        source = Path(self.temporary.name) / "source.bin"
        source.write_bytes(b"frozen")
        link = Path(self.temporary.name) / "link.bin"
        link.symlink_to(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(target.BundleError, "must not be a symlink"):
            target._read_verified(
                link, label="synthetic input", sha256=digest, size=6
            )


if __name__ == "__main__":
    unittest.main()
