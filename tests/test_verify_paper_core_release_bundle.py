"""Structural tests for the streaming paper-core release verifier."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_paper_core_release_bundle as builder
from scripts import verify_paper_core_release_bundle as target


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


class PaperCoreBundleVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = json.loads(
            (PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json")
            .read_text(encoding="utf-8")
        )
        total = 0
        self.checkpoint_raw = {}
        for index, record in enumerate(source["records"]):
            raw = f"synthetic-checkpoint-{index}".encode("ascii")
            self.checkpoint_raw[record["public_file_name"]] = raw
            public = record["public_identity"]
            public["bytes"] = len(raw)
            public["sha256"] = hashlib.sha256(raw).hexdigest()
            public["tensor_fingerprint"]["sha256"] = hashlib.sha256(
                b"tensor-" + raw
            ).hexdigest()
            public["tensor_fingerprint"]["tensor_count"] = index + 1
            total += len(raw)
        source["public_total_bytes"] = total
        source["identity_audit"]["public_checkpoint_bytes"] = total
        self.spec = source
        self.spec_raw = _canonical(source)
        self.spec_path = self.root / "spec.json"
        self.spec_path.write_bytes(self.spec_raw)
        self.commit = target.source_commit(PROJECT_ROOT)
        self.original_git_blob = target._git_blob

    def _verify(self, archive, checksum):
        def committed_blob(project_root, commit, relative_path):
            if relative_path == "artifacts/paper_core_checkpoint_release_spec.json":
                return self.spec_raw
            return self.original_git_blob(project_root, commit, relative_path)

        with mock.patch.object(target, "_git_blob", committed_blob):
            return target.verify(archive, checksum, self.spec_path)

    def _write_bundle(
        self,
        *,
        self_consistent_checkpoint_tamper=False,
        static_readme_tamper=False,
        manifest_commit=None,
        pax_metadata=False,
    ):
        static = {
            "README.md": target.expected_bundle_readme(self.spec, self.commit),
            "LICENSE": target.expected_license(PROJECT_ROOT, self.commit),
            "THIRD_PARTY_NOTICES.md": target.expected_third_party_notice(
                PROJECT_ROOT, self.commit
            ),
            target.SPEC_MEMBER: self.spec_raw,
            target.AUDIT_MEMBER: (
                PROJECT_ROOT / "artifacts" / "paper_core_input_audit.json"
            ).read_bytes(),
        }
        checkpoints = dict(self.checkpoint_raw)
        if static_readme_tamper:
            static["README.md"] += b"self-consistent tamper\n"
        if self_consistent_checkpoint_tamper:
            first = self.spec["records"][0]["public_file_name"]
            checkpoints[first] = b"self-consistent-but-not-pinned"
        payload = {
            **static,
            **{f"checkpoints/{name}": raw for name, raw in checkpoints.items()},
        }
        identities = {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in payload.items()
        }
        manifest = builder._build_manifest(
            self.spec,
            self.commit if manifest_commit is None else manifest_commit,
            identities,
        )
        manifest_raw = _canonical(manifest)
        checksummed = {
            **identities,
            "BUNDLE_MANIFEST.json": {
                "bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
        }
        sums_raw = "".join(
            f"{identity['sha256']}  {name}\n"
            for name, identity in sorted(checksummed.items())
        ).encode("ascii")
        archive = self.root / self.spec["archive_name"]
        with archive.open("wb") as raw_archive:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_archive, mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    root = self.spec["archive_root"]
                    for directory in (
                        root,
                        f"{root}/checkpoints",
                        f"{root}/metadata",
                    ):
                        tar.addfile(builder._tar_info(directory, directory=True))
                    builder._add_bytes(
                        tar, f"{root}/BUNDLE_MANIFEST.json", manifest_raw
                    )
                    builder._add_bytes(tar, f"{root}/SHA256SUMS", sums_raw)
                    for name in (
                        "README.md",
                        "LICENSE",
                        "THIRD_PARTY_NOTICES.md",
                        target.SPEC_MEMBER,
                        target.AUDIT_MEMBER,
                    ):
                        member_name = f"{root}/{name}"
                        if pax_metadata and name == "README.md":
                            info = builder._tar_info(
                                member_name, size=len(static[name])
                            )
                            info.pax_headers = {"comment": "/secret/private/path"}
                            tar.addfile(info, io.BytesIO(static[name]))
                        else:
                            builder._add_bytes(tar, member_name, static[name])
                    for record in self.spec["records"]:
                        name = record["public_file_name"]
                        builder._add_bytes(
                            tar, f"{root}/checkpoints/{name}", checkpoints[name]
                        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = self.root / f"{archive.name}.sha256"
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
        return archive, checksum

    def _rewrite_uncompressed_tar(self, archive, checksum, transform):
        raw = gzip.decompress(archive.read_bytes())
        changed = transform(raw)
        with archive.open("wb") as output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=output, mtime=0
            ) as compressed:
                compressed.write(changed)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")

    def test_streaming_verifier_accepts_exact_pinned_bundle(self):
        archive, checksum = self._write_bundle()
        report = self._verify(archive, checksum)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checkpoint_count"], 12)
        self.assertEqual(
            report["public_checkpoint_bytes"], self.spec["public_total_bytes"]
        )

    def test_self_consistent_checkpoint_tamper_is_rejected_by_pinned_spec(self):
        archive, checksum = self._write_bundle(
            self_consistent_checkpoint_tamper=True
        )
        with self.assertRaisesRegex(
            target.VerificationError, "member size changed|checkpoint identity"
        ):
            self._verify(archive, checksum)

    def test_outer_checksum_is_verified_before_tar_processing(self):
        archive, checksum = self._write_bundle()
        checksum.write_text(f"{'0' * 64}  {archive.name}\n", encoding="ascii")
        with self.assertRaisesRegex(target.VerificationError, "Outer archive"):
            self._verify(archive, checksum)

    def test_manifest_source_commit_must_match_verifier_checkout(self):
        archive, checksum = self._write_bundle(manifest_commit="a" * 40)
        with self.assertRaisesRegex(
            target.VerificationError, "Invalid source repository identity"
        ):
            self._verify(archive, checksum)

    def test_self_consistent_static_file_tamper_is_rejected(self):
        archive, checksum = self._write_bundle(static_readme_tamper=True)
        with self.assertRaisesRegex(
            target.VerificationError, "member size changed|static file differs"
        ):
            self._verify(archive, checksum)

    def test_hidden_pax_metadata_is_rejected(self):
        archive, checksum = self._write_bundle(pax_metadata=True)
        with self.assertRaisesRegex(
            target.VerificationError, "tar header|order/name"
        ):
            self._verify(archive, checksum)

    def test_custom_spec_not_bound_to_source_commit_is_rejected(self):
        archive, checksum = self._write_bundle()
        with self.assertRaisesRegex(
            target.VerificationError, "specification differs"
        ):
            target.verify(archive, checksum, self.spec_path)

    def test_archive_path_swap_during_verification_is_rejected(self):
        archive, checksum = self._write_bundle()
        replacement = self.root / "replacement.tar.gz"
        shutil.copyfile(archive, replacement)
        with replacement.open("ab") as stream:
            stream.write(b"replacement")
        original_hash = target._sha256_stream
        calls = 0

        def hash_and_swap(stream):
            nonlocal calls
            digest = original_hash(stream)
            calls += 1
            if calls == 1:
                os.replace(replacement, archive)
            return digest

        with mock.patch.object(target, "_sha256_stream", hash_and_swap):
            with self.assertRaisesRegex(
                target.VerificationError, "Archive (path )?changed"
            ):
                self._verify(archive, checksum)

    def test_raw_trailer_after_valid_gzip_is_rejected(self):
        archive, checksum = self._write_bundle()
        with archive.open("ab") as stream:
            stream.write(b"UNTRACKED-TRAILER:/secret/path")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
        with self.assertRaisesRegex(
            target.VerificationError, "raw trailer"
        ):
            self._verify(archive, checksum)

    def test_noncanonical_gzip_header_is_rejected(self):
        archive, checksum = self._write_bundle()
        raw = gzip.decompress(archive.read_bytes())
        with archive.open("wb") as output:
            with gzip.GzipFile(
                filename="gpuadmin-private-path",
                mode="wb",
                fileobj=output,
                mtime=12345,
                compresslevel=9,
            ) as compressed:
                compressed.write(raw)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
        with self.assertRaisesRegex(
            target.VerificationError, "Non-canonical gzip header"
        ):
            self._verify(archive, checksum)

    def test_uncompressed_data_after_tar_eof_is_rejected(self):
        archive, checksum = self._write_bundle()
        self._rewrite_uncompressed_tar(
            archive,
            checksum,
            lambda raw: raw + b"UNTRACKED-IN-GZIP:/secret/private/path",
        )
        with self.assertRaisesRegex(
            target.VerificationError, "tar EOF|canonical"
        ):
            self._verify(archive, checksum)

    def test_nonzero_member_padding_is_rejected(self):
        archive, checksum = self._write_bundle()

        def hide_in_readme_padding(raw):
            changed = bytearray(raw)
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as stream:
                member = stream.getmember(
                    f"{self.spec['archive_root']}/README.md"
                )
            start = member.offset_data + member.size
            padding = (-member.size) % tarfile.BLOCKSIZE
            marker = b"/home/hidden"
            self.assertGreaterEqual(padding, len(marker))
            changed[start:start + len(marker)] = marker
            return bytes(changed)

        self._rewrite_uncompressed_tar(
            archive, checksum, hide_in_readme_padding
        )
        with self.assertRaisesRegex(
            target.VerificationError, "Non-zero tar member padding"
        ):
            self._verify(archive, checksum)


if __name__ == "__main__":
    unittest.main()
