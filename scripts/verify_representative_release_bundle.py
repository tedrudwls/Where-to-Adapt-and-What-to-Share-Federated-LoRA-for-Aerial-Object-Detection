#!/usr/bin/env python3
"""Fail-closed structural verifier for the representative release archive.

Verify the separately published outer SHA-256 file before reading tar members.
This utility never deserializes the PyTorch checkpoint.  Model evaluation must
additionally verify the code-pinned public checkpoint identity before loading it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence


sys.dont_write_bytecode = True

ARCHIVE_NAME = "fedlora-representative-replay-seed42-v1.0.0.tar.gz"
ROOT = "representative-replay"
CHECKPOINT_NAME = "seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt"
REPLAY_NAME = "seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json"
REPLAY_SHA256 = "ffe7b2932dfcfb0027a8e607abb3d8d5c8a5241d549b280524b77be5f3b44c88"
REPLAY_BYTES = 472024
HISTORICAL_CHECKPOINT_SHA256 = (
    "3ed025419506009465add698da75fef941c20c0b16eb347bb224820781b9cac4"
)
HISTORICAL_CHECKPOINT_BYTES = 3316516
PUBLIC_CHECKPOINT_SHA256 = (
    "391205473ad8de24af56ba1b566e54a6f305dd0b79806cb468583e84e464fa14"
)
PUBLIC_CHECKPOINT_BYTES = 3316452
PUBLIC_TENSOR_FINGERPRINT_SHA256 = (
    "b42e1e811238caf1ec76e788547dbcf46d8f390b3b0e63864cf3d303e88c6d4f"
)
PUBLIC_TENSOR_COUNT = 231
HISTORICAL_SPLIT_SHA256 = (
    "74a45a37f4b05c474564993ff15f5548875f3e767abc19a0d2d30f195e1754c5"
)
PRETRAINED_SHA256 = (
    "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
)
DATASET_DOI = "10.17632/cd5z895tr2.1"
TEST_IMAGE_TREE_SHA256 = (
    "42776a3cfca2f66df7efc1215eb7f64e4d92f445d9e917e4d5f8d0909c50e411"
)
SOURCE_REPOSITORY = (
    "https://github.com/tedrudwls/"
    "Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection"
)
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_MEMBER_BYTES = 10 * 1024 * 1024
MAX_TAR_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 15 * 1024 * 1024
EXPECTED_FILES = {
    f"{ROOT}/BUNDLE_MANIFEST.json",
    f"{ROOT}/SHA256SUMS",
    f"{ROOT}/README.md",
    f"{ROOT}/LICENSE",
    f"{ROOT}/THIRD_PARTY_NOTICES.md",
    f"{ROOT}/checkpoint/{CHECKPOINT_NAME}",
    f"{ROOT}/metadata/{REPLAY_NAME}",
}
EXPECTED_DIRECTORIES = {ROOT, f"{ROOT}/checkpoint", f"{ROOT}/metadata"}
FORBIDDEN_TEXT = (b"/home/", b"/Users/", b"gpuadmin", b"file://")


class VerificationError(RuntimeError):
    """The downloaded archive does not satisfy the frozen public layout."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    if not value or "\\" in value:
        raise VerificationError(f"Unsafe tar member name: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise VerificationError(f"Unsafe tar member name: {value!r}")
    return pure.as_posix().rstrip("/")


def _parse_outer_checksum(path: Path, archive_name: str) -> str:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"Cannot read outer checksum: {error}") from error
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", text)
    if not match or match.group(2) != archive_name:
        raise VerificationError("Outer checksum file has an unexpected format/name")
    return match.group(1)


def _parse_internal_checksums(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise VerificationError("Internal SHA256SUMS is not ASCII") from error
    records = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise VerificationError(f"Malformed SHA256SUMS row: {line!r}")
        name = _safe_name(match.group(2))
        if name in records:
            raise VerificationError(f"Duplicate SHA256SUMS path: {name}")
        records[name] = match.group(1)
    expected = {
        "BUNDLE_MANIFEST.json", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
        f"checkpoint/{CHECKPOINT_NAME}", f"metadata/{REPLAY_NAME}",
    }
    if set(records) != expected:
        raise VerificationError("Internal SHA256SUMS file set changed")
    return records


def _validate_manifest(raw: bytes, files: dict[str, bytes]) -> dict:
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot parse BUNDLE_MANIFEST.json: {error}") from error
    if not isinstance(manifest, dict):
        raise VerificationError("BUNDLE_MANIFEST.json must be an object")
    expected_top_keys = {
        "schema_version", "bundle_id", "release_version", "experiment_id",
        "method", "source_repository", "files",
        "historical_split_manifest_sha256", "external_inputs", "expected_metrics",
    }
    if set(manifest) != expected_top_keys:
        raise VerificationError("Bundle manifest top-level schema changed")
    checks = {
        "schema_version": (manifest.get("schema_version"), 1),
        "bundle_id": (manifest.get("bundle_id"), "seed42-fedlora-a-r8-a04-replay"),
        "release_version": (manifest.get("release_version"), "1.0.0"),
        "experiment_id": (
            manifest.get("experiment_id"), "seed_42/fl_fedsa_lora_r8_a0.4"
        ),
        "historical_split_manifest_sha256": (
            manifest.get("historical_split_manifest_sha256"),
            HISTORICAL_SPLIT_SHA256,
        ),
    }
    mismatches = {
        key: {"bundle": left, "required": right}
        for key, (left, right) in checks.items() if left != right
    }
    if mismatches:
        raise VerificationError(
            "Bundle manifest identity changed: " + json.dumps(mismatches, sort_keys=True)
        )
    if manifest.get("method") != "FedLoRA-A (Share-A / local B)":
        raise VerificationError("Bundle method identity changed")
    source = manifest["source_repository"]
    if (
        not isinstance(source, dict)
        or set(source) != {"url", "commit"}
        or source.get("url") != SOURCE_REPOSITORY
        or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))
    ):
        raise VerificationError("Bundle source repository identity is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 5:
        raise VerificationError("Bundle manifest must describe five payload files")
    by_path = {}
    expected_roles = {
        f"checkpoint/{CHECKPOINT_NAME}": "sanitized_validation_selected_checkpoint",
        f"metadata/{REPLAY_NAME}": "path_free_test_replay_manifest",
        "README.md": "bundle_readme",
        "LICENSE": "license",
        "THIRD_PARTY_NOTICES.md": "third_party_notices",
    }
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Bundle file record is not an object")
        name = _safe_name(str(record.get("path", "")))
        if name in by_path:
            raise VerificationError(f"Duplicate bundle file record: {name}")
        if name not in files:
            raise VerificationError(f"Bundle file record targets a missing file: {name}")
        if int(record.get("bytes", -1)) != len(files[name]):
            raise VerificationError(f"Bundle manifest size mismatch: {name}")
        if record.get("sha256") != sha256_bytes(files[name]):
            raise VerificationError(f"Bundle manifest SHA-256 mismatch: {name}")
        expected_keys = {"path", "role", "bytes", "sha256"}
        if name == f"checkpoint/{CHECKPOINT_NAME}":
            expected_keys |= {
                "historical_source", "path_replacements", "tensor_fingerprint"
            }
        if set(record) != expected_keys:
            raise VerificationError(f"Bundle file-record schema changed: {name}")
        if record.get("role") != expected_roles.get(name):
            raise VerificationError(f"Bundle file role changed: {name}")
        by_path[name] = record
    expected_records = {
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
        f"checkpoint/{CHECKPOINT_NAME}", f"metadata/{REPLAY_NAME}",
    }
    if set(by_path) != expected_records:
        raise VerificationError("Bundle manifest payload file set changed")
    checkpoint = by_path[f"checkpoint/{CHECKPOINT_NAME}"]
    if (
        checkpoint.get("bytes") != PUBLIC_CHECKPOINT_BYTES
        or checkpoint.get("sha256") != PUBLIC_CHECKPOINT_SHA256
    ):
        raise VerificationError("Public checkpoint identity changed")
    if checkpoint.get("historical_source") != {
        "bytes": HISTORICAL_CHECKPOINT_BYTES,
        "sha256": HISTORICAL_CHECKPOINT_SHA256,
    }:
        raise VerificationError("Historical checkpoint identity changed")
    expected_pointers = {
        "/architecture/model_weight_path", "/experiment/model_weights"
    }
    replacements = checkpoint.get("path_replacements")
    if (
        not isinstance(replacements, list)
        or len(replacements) != 2
        or {row.get("json_pointer") for row in replacements} != expected_pointers
        or any(row.get("public_value") != "external/rtdetr-l.pt" for row in replacements)
        or any(
            not isinstance(row, dict)
            or set(row) != {
                "json_pointer", "historical_value_sha256", "public_value"
            }
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("historical_value_sha256", ""))
            )
            for row in replacements
        )
    ):
        raise VerificationError("Checkpoint path-replacement record changed")
    tensor = checkpoint.get("tensor_fingerprint")
    if (
        not isinstance(tensor, dict)
        or tensor.get("algorithm")
        != "recursive_path_dtype_shape_raw_bytes_sha256_v1"
        or not isinstance(tensor.get("tensor_count"), int)
        or tensor["tensor_count"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(tensor.get("sha256", "")))
    ):
        raise VerificationError("Checkpoint tensor fingerprint is invalid")
    if (
        tensor.get("sha256") != PUBLIC_TENSOR_FINGERPRINT_SHA256
        or tensor.get("tensor_count") != PUBLIC_TENSOR_COUNT
    ):
        raise VerificationError("Public checkpoint tensor fingerprint changed")
    external = manifest.get("external_inputs", {})
    if not isinstance(external, dict) or set(external) != {
        "pretrained_model", "dataset"
    }:
        raise VerificationError("External input schema changed")
    pretrained = external.get("pretrained_model", {})
    dataset = external.get("dataset", {})
    if (
        not isinstance(pretrained, dict)
        or set(pretrained) != {"included", "file_name", "sha256"}
        or pretrained.get("included") is not False
        or pretrained.get("file_name") != "rtdetr-l.pt"
        or pretrained.get("sha256") != PRETRAINED_SHA256
        or not isinstance(dataset, dict)
        or set(dataset) != {
            "included", "doi", "test_images", "test_image_tree_sha256"
        }
        or dataset.get("included") is not False
        or dataset.get("doi") != DATASET_DOI
        or int(dataset.get("test_images", -1)) != 2241
        or dataset.get("test_image_tree_sha256") != TEST_IMAGE_TREE_SHA256
    ):
        raise VerificationError("External input contract changed")
    expected_metrics = manifest.get("expected_metrics")
    if expected_metrics != {
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
    }:
        raise VerificationError("Bundle expected metrics changed")
    return manifest


def verify(archive: Path, checksum: Path) -> dict:
    archive_input = archive.expanduser()
    checksum_input = checksum.expanduser()
    if archive_input.is_symlink() or checksum_input.is_symlink():
        raise VerificationError("Archive/checksum must not be symlinks")
    archive = archive_input.resolve(strict=True)
    checksum = checksum_input.resolve(strict=True)
    if archive.name != ARCHIVE_NAME:
        raise VerificationError(f"Unexpected archive name: {archive.name}")
    if not archive.is_file() or not checksum.is_file():
        raise VerificationError("Archive/checksum must be regular files")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise VerificationError("Archive is unexpectedly large")
    expected_outer = _parse_outer_checksum(checksum, archive.name)
    actual_outer = sha256_file(archive)
    if actual_outer != expected_outer:
        raise VerificationError(
            f"Outer archive SHA-256 mismatch: expected={expected_outer}, actual={actual_outer}"
        )

    with gzip.open(archive, mode="rb") as compressed:
        tar_raw = compressed.read(MAX_TAR_BYTES + 1)
    if len(tar_raw) > MAX_TAR_BYTES:
        raise VerificationError("Decompressed tar stream is unexpectedly large")

    member_names = set()
    raw_files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_raw), mode="r:") as tar:
        members = tar.getmembers()
        expected_members = EXPECTED_FILES | EXPECTED_DIRECTORIES
        if len(members) != len(expected_members):
            raise VerificationError("Tar member count changed")
        total_payload = 0
        for member in members:
            name = _safe_name(member.name)
            if name in member_names:
                raise VerificationError(f"Duplicate tar member: {name}")
            member_names.add(name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise VerificationError(f"Unsafe tar member type: {name}")
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise VerificationError(f"Invalid tar member size/type: {name}")
            total_payload += member.size
            if total_payload > MAX_PAYLOAD_BYTES:
                raise VerificationError("Tar payload is unexpectedly large")
        if member_names != expected_members:
            raise VerificationError("Tar member allowlist changed")
        for member in members:
            name = _safe_name(member.name)
            if member.isdir():
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise VerificationError(f"Cannot read tar member: {name}")
            raw = stream.read(MAX_MEMBER_BYTES + 1)
            if len(raw) != member.size:
                raise VerificationError(f"Tar member byte count mismatch: {name}")
            raw_files[name] = raw
    relative = {
        name[len(ROOT) + 1:]: raw for name, raw in raw_files.items()
    }
    checksums = _parse_internal_checksums(relative["SHA256SUMS"])
    for name, digest in checksums.items():
        if sha256_bytes(relative[name]) != digest:
            raise VerificationError(f"Internal SHA-256 mismatch: {name}")
    replay = relative[f"metadata/{REPLAY_NAME}"]
    if len(replay) != REPLAY_BYTES or sha256_bytes(replay) != REPLAY_SHA256:
        raise VerificationError("Public replay manifest identity changed")
    for name, raw in relative.items():
        if name.startswith("checkpoint/"):
            continue
        if any(value in raw for value in FORBIDDEN_TEXT) or re.search(
            rb"(?<![A-Za-z])[A-Za-z]:[\\/]", raw
        ):
            raise VerificationError(f"Private-path marker found in {name}")
    manifest = _validate_manifest(relative["BUNDLE_MANIFEST.json"], relative)
    checkpoint_record = next(
        row for row in manifest["files"]
        if row["path"] == f"checkpoint/{CHECKPOINT_NAME}"
    )
    return {
        "status": "pass",
        "archive": {"file_name": archive.name, "bytes": archive.stat().st_size,
                    "sha256": actual_outer},
        "public_checkpoint": {
            "file_name": CHECKPOINT_NAME,
            "bytes": checkpoint_record["bytes"],
            "sha256": checkpoint_record["sha256"],
            "tensor_fingerprint": checkpoint_record["tensor_fingerprint"],
        },
        "public_replay_manifest": {
            "file_name": REPLAY_NAME, "bytes": REPLAY_BYTES, "sha256": REPLAY_SHA256,
        },
        "source_commit": manifest["source_repository"]["commit"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = verify(args.archive, args.checksum)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
