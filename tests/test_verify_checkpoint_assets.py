import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_checkpoint_assets import load_index, main


class CheckpointAssetVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.relative = "results/seed_42/weights/best_federated.pt"
        self.checkpoint = self.project / self.relative
        self.checkpoint.parent.mkdir(parents=True)
        self.payload = b"synthetic checkpoint bytes"
        self.checkpoint.write_bytes(self.payload)
        self.index = self.root / "checkpoint_index.json"
        self.record = {
            "historical_project_relative_path": self.relative,
            "release_asset_name": "seed_42__best_federated.pt",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.write_index()

    def write_index(self):
        self.index.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_count": 1,
                    "total_bytes": self.record["bytes"],
                    "records": [self.record],
                }
            ),
            encoding="utf-8",
        )

    def invoke(self, output_format="text"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "--project-dir",
                    str(self.project),
                    "--index",
                    str(self.index),
                    "--format",
                    output_format,
                ]
            )
        return code, output.getvalue()

    def test_matching_checkpoint_and_json_output(self):
        code, output = self.invoke("json")
        self.assertEqual(code, 0)
        result = json.loads(output)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["verified_bytes"], len(self.payload))
        self.assertEqual(result["results"][0]["status"], "ok")

    def test_missing_file(self):
        self.checkpoint.unlink()
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("missing:", output)

    def test_size_mismatch(self):
        self.checkpoint.write_bytes(self.payload + b"!")
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("size_mismatch:", output)

    def test_sha256_mismatch(self):
        self.checkpoint.write_bytes(b"X" + self.payload[1:])
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("sha256_mismatch:", output)

    def test_rejects_parent_path(self):
        self.record["historical_project_relative_path"] = "../outside.pt"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "unsafe relative path"):
            load_index(self.index)

    def test_rejects_symlink_outside_project(self):
        outside = self.root / "outside.pt"
        outside.write_bytes(self.payload)
        self.checkpoint.unlink()
        self.checkpoint.symlink_to(outside)
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("path_outside_project:", output)


if __name__ == "__main__":
    unittest.main()
