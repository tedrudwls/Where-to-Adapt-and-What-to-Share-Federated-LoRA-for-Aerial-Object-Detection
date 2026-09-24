#!/usr/bin/env python3
"""Fail-closed streaming verifier for the 12-checkpoint paper-core archive.

The outer checksum is verified before the gzip stream is opened. Tar members
are then hashed incrementally; checkpoint bytes are never accumulated in
memory or deserialized by this structural verifier. The committed, pinned
release specification is the independent trust root for all 12 public files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Optional, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json"
SOURCE_REPOSITORY = (
    "https://github.com/tedrudwls/"
    "Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection"
)
SPEC_MEMBER = "metadata/paper_core_checkpoint_release_spec.json"
AUDIT_MEMBER = "metadata/paper_core_input_audit.json"
SMALL_FILE_LIMIT = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 140 * 1024 * 1024
MAX_PAYLOAD_BYTES = 600 * 1024 * 1024
MAX_GZIP_UNCOMPRESSED_BYTES = MAX_PAYLOAD_BYTES + 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_TEXT = (b"/home/", b"/Users/", b"gpuadmin", b"file://")
CANONICAL_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"


class VerificationError(RuntimeError):
    """The archive does not satisfy the independently pinned release contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class _SingleGzipReader:
    """Bounded single-member gzip reader that also hashes the raw stream."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.digest = hashlib.sha256()
        self.buffer = bytearray()
        self.uncompressed_bytes = 0
        self.eof = False
        self.header_checked = False
        self.pending = b""

    def _pump(self, output_limit: int) -> None:
        if self.eof:
            return
        if not self.pending:
            chunk = self.stream.read(4 * 1024 * 1024)
            if not chunk:
                raise VerificationError("Truncated gzip stream")
            if not self.header_checked:
                if not chunk.startswith(CANONICAL_GZIP_HEADER):
                    raise VerificationError("Non-canonical gzip header")
                self.header_checked = True
            self.digest.update(chunk)
            self.pending = chunk
        try:
            output = self.decoder.decompress(self.pending, output_limit)
        except zlib.error as error:
            raise VerificationError(f"Invalid gzip stream: {error}") from error
        self.pending = self.decoder.unconsumed_tail
        self.buffer.extend(output)
        self.uncompressed_bytes += len(output)
        if self.uncompressed_bytes > MAX_GZIP_UNCOMPRESSED_BYTES:
            raise VerificationError("Uncompressed gzip stream exceeds size cap")
        if len(self.buffer) > (4 * 1024 * 1024):
            raise VerificationError("Gzip decode buffer exceeded its bound")
        if self.decoder.eof:
            if self.decoder.unused_data or self.pending or self.stream.read(1):
                raise VerificationError(
                    "Archive contains concatenated gzip data or a raw trailer"
                )
            self.eof = True

    def read_exact(self, size: int) -> bytes:
        while len(self.buffer) < size and not self.eof:
            self._pump(min(4 * 1024 * 1024, size - len(self.buffer)))
        if len(self.buffer) < size:
            raise VerificationError("Truncated uncompressed tar stream")
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value

    def finish(self) -> str:
        while not self.eof:
            self._pump(4 * 1024 * 1024)
        if self.buffer:
            raise VerificationError("Untracked data follows canonical tar EOF")
        return self.digest.hexdigest()


def source_commit(project_root: Path = PROJECT_ROOT) -> str:
    """Return the immutable commit that the verifier is expected to represent."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError("Cannot resolve verifier source commit") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise VerificationError("Verifier source commit is not an immutable SHA-1")
    return commit


def _git_blob(project_root: Path, commit: str, relative_path: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise VerificationError("Invalid expected source commit")
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"], cwd=project_root
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(
            f"Cannot read {relative_path} from source commit {commit}"
        ) from error


def expected_bundle_readme(spec: Mapping[str, Any], commit: str) -> bytes:
    return (
        "# FedLoRA paper-core checkpoints\n\n"
        "This archive contains 12 path-sanitized, validation-selected federated "
        "evaluation checkpoints: FL Full FT, FedLoRA-AB, FedLoRA-A and "
        "FedLoRA-B for paired seeds 42, 43 and 44 under Dirichlet alpha=0.4.\n\n"
        "The archive excludes AOD-4 data, historical result JSONs, split "
        "manifests and the upstream `rtdetr-l.pt` pretrained weight. Obtain the "
        "external inputs separately and verify their documented hashes. These "
        "checkpoints support inference/evaluation and do not provide exact "
        "training resumption.\n\n"
        f"Release ID: `{spec['release_id']}`\n\n"
        f"Source repository: {SOURCE_REPOSITORY}\n\n"
        f"Source commit: `{commit}`\n\n"
        "Verify the separately distributed outer `.sha256` file before opening "
        "the archive. Then run `scripts/verify_paper_core_release_bundle.py` at "
        "the source commit above; it streams every member and binds all 12 "
        "checkpoint identities to the committed release specification. Never "
        "deserialize a checkpoint before its published SHA-256 is verified.\n"
    ).encode("utf-8")


def expected_license(project_root: Path, commit: str) -> bytes:
    return _git_blob(project_root, commit, "LICENSE")


def expected_third_party_notice(project_root: Path, commit: str) -> bytes:
    try:
        text = _git_blob(
            project_root, commit, "THIRD_PARTY_NOTICES.md"
        ).decode("utf-8")
    except UnicodeError as error:
        raise VerificationError("THIRD_PARTY_NOTICES.md is not UTF-8") from error
    relative_link = "[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)"
    immutable_url = (
        f"[`docs/CHECKPOINTS.md`]({SOURCE_REPOSITORY}/blob/{commit}/"
        "docs/CHECKPOINTS.md)"
    )
    if text.count(relative_link) != 1:
        raise VerificationError(
            "Third-party notice checkpoint link changed unexpectedly"
        )
    return text.replace(relative_link, immutable_url).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        raise VerificationError(f"Unsafe archive path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise VerificationError(f"Unsafe archive path: {value!r}")
    return pure.as_posix().rstrip("/")


def load_pinned_spec(path: Path) -> tuple[dict, bytes]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise VerificationError("Release specification must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        raw = resolved.read_bytes()
        spec = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read release specification: {error}") from error
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise VerificationError("Release specification must use schema_version=1")
    archive_name = spec.get("archive_name")
    archive_root = spec.get("archive_root")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or "\\" in archive_name
        or not archive_name.endswith(".tar.gz")
        or not isinstance(archive_root, str)
        or _safe_relative_name(archive_root) != archive_root
        or len(PurePosixPath(archive_root).parts) != 1
    ):
        raise VerificationError("Release specification has an unsafe archive layout")
    if spec.get("status") != "public_identities_pinned":
        raise VerificationError("Release specification has no pinned public identities")
    if spec.get("expected_record_count") != 12:
        raise VerificationError("Release specification must contain 12 checkpoints")
    records = spec.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise VerificationError("Release specification record count changed")
    experiment_ids = set()
    file_names = set()
    public_total = 0
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Malformed release-spec record")
        experiment_id = record.get("experiment_id")
        file_name = record.get("public_file_name")
        public = record.get("public_identity")
        if (
            not isinstance(experiment_id, str)
            or experiment_id in experiment_ids
            or not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or "\\" in file_name
            or "/" in file_name
            or file_name in file_names
            or not isinstance(public, dict)
        ):
            raise VerificationError("Malformed or duplicate release-spec identity")
        tensor = public.get("tensor_fingerprint")
        if (
            set(public) != {
                "bytes", "sha256", "tensor_fingerprint",
                "path_replacements", "residual_absolute_path_count",
            }
            or isinstance(public.get("bytes"), bool)
            or not isinstance(public.get("bytes"), int)
            or not 0 < public["bytes"] <= MAX_MEMBER_BYTES
            or not SHA256_PATTERN.fullmatch(str(public.get("sha256", "")))
            or public.get("residual_absolute_path_count") != 0
            or not isinstance(tensor, dict)
            or tensor.get("algorithm")
            != "recursive_path_dtype_shape_raw_bytes_sha256_v1"
            or isinstance(tensor.get("tensor_count"), bool)
            or not isinstance(tensor.get("tensor_count"), int)
            or tensor["tensor_count"] <= 0
            or not SHA256_PATTERN.fullmatch(str(tensor.get("sha256", "")))
        ):
            raise VerificationError(f"Malformed public identity: {experiment_id}")
        replacements = public.get("path_replacements")
        expected_pointers = {
            "/architecture/model_weight_path",
            "/experiment/model_weights",
        }
        if (
            not isinstance(replacements, list)
            or len(replacements) != 2
            or any(not isinstance(row, dict) for row in replacements)
            or {row.get("json_pointer") for row in replacements} != expected_pointers
            or any(
                set(row) != {
                    "json_pointer", "historical_value_sha256", "public_value"
                }
                or row.get("public_value") != "external/rtdetr-l.pt"
                or not SHA256_PATTERN.fullmatch(
                    str(row.get("historical_value_sha256", ""))
                )
                for row in replacements
            )
        ):
            raise VerificationError(f"Malformed path replacements: {experiment_id}")
        experiment_ids.add(experiment_id)
        file_names.add(file_name)
        public_total += public["bytes"]
    if public_total != spec.get("public_total_bytes"):
        raise VerificationError("Release-spec public byte total changed")
    audit_receipt = spec.get("identity_audit")
    if (
        not isinstance(audit_receipt, dict)
        or audit_receipt.get("status") != "pass"
        or audit_receipt.get("policy")
        != "paper_core_12_checkpoint_read_only_prepublication_audit"
        or isinstance(audit_receipt.get("report_bytes"), bool)
        or not isinstance(audit_receipt.get("report_bytes"), int)
        or audit_receipt["report_bytes"] <= 0
        or not SHA256_PATTERN.fullmatch(
            str(audit_receipt.get("report_sha256", ""))
        )
        or audit_receipt.get("protected_file_count") != 14
        or audit_receipt.get("historical_checkpoint_bytes")
        != spec.get("historical_total_bytes")
        or audit_receipt.get("public_checkpoint_bytes") != public_total
    ):
        raise VerificationError("Release-spec identity-audit receipt changed")
    if any(marker in raw for marker in FORBIDDEN_TEXT):
        raise VerificationError("Release specification contains a private path marker")
    return spec, raw


def _parse_outer_checksum(path: Path, archive_name: str) -> str:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise VerificationError("Outer checksum must not be a symlink")
    try:
        text = candidate.resolve(strict=True).read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"Cannot read outer checksum: {error}") from error
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", text)
    if not match or match.group(2) != archive_name:
        raise VerificationError("Outer checksum has an unexpected format or file name")
    return match.group(1)


def _expected_relative_files(spec: Mapping[str, Any]) -> set[str]:
    return {
        "BUNDLE_MANIFEST.json",
        "SHA256SUMS",
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        SPEC_MEMBER,
        AUDIT_MEMBER,
        *{
            f"checkpoints/{record['public_file_name']}"
            for record in spec["records"]
        },
    }


def _expected_member_order(spec: Mapping[str, Any]) -> list[str]:
    root = spec["archive_root"]
    return [
        root,
        f"{root}/checkpoints",
        f"{root}/metadata",
        f"{root}/BUNDLE_MANIFEST.json",
        f"{root}/SHA256SUMS",
        f"{root}/README.md",
        f"{root}/LICENSE",
        f"{root}/THIRD_PARTY_NOTICES.md",
        f"{root}/{SPEC_MEMBER}",
        f"{root}/{AUDIT_MEMBER}",
        *[
            f"{root}/checkpoints/{record['public_file_name']}"
            for record in spec["records"]
        ],
    ]


def _parse_canonical_tar_stream(
    raw_stream: BinaryIO,
    spec: Mapping[str, Any],
    spec_raw: bytes,
    expected_static: Mapping[str, bytes],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, bytes]]:
    """Parse exactly the builder's physical tar framing over one gzip member."""

    root = spec["archive_root"]
    order = _expected_member_order(spec)
    directories = {root, f"{root}/checkpoints", f"{root}/metadata"}
    known_sizes = {
        **{name: 0 for name in directories},
        **{
            f"{root}/{name}": len(raw)
            for name, raw in expected_static.items()
        },
        f"{root}/{SPEC_MEMBER}": len(spec_raw),
        f"{root}/{AUDIT_MEMBER}": spec["identity_audit"]["report_bytes"],
        **{
            f"{root}/checkpoints/{record['public_file_name']}":
            record["public_identity"]["bytes"]
            for record in spec["records"]
        },
    }
    identities: dict[str, dict[str, Any]] = {}
    small_files: dict[str, bytes] = {}
    total_payload = 0
    tar_offset = 0
    reader = _SingleGzipReader(raw_stream)

    for expected_name in order:
        header = reader.read_exact(tarfile.BLOCKSIZE)
        tar_offset += tarfile.BLOCKSIZE
        if header == (b"\0" * tarfile.BLOCKSIZE):
            raise VerificationError("Tar EOF appeared before the member allowlist")
        try:
            member = tarfile.TarInfo.frombuf(
                header, encoding="utf-8", errors="surrogateescape"
            )
            canonical_header = member.tobuf(
                format=tarfile.PAX_FORMAT,
                encoding="utf-8",
                errors="surrogateescape",
            )
        except (tarfile.TarError, UnicodeError, ValueError) as error:
            raise VerificationError(f"Invalid tar header: {error}") from error
        if canonical_header != header:
            raise VerificationError(f"Non-canonical tar header: {expected_name}")
        if member.name != expected_name:
            raise VerificationError(
                f"Tar member order/name changed: expected={expected_name!r}, "
                f"actual={member.name!r}"
            )
        is_directory = expected_name in directories
        expected_mode = 0o755 if is_directory else 0o644
        expected_type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
        if (
            member.type != expected_type
            or member.uid != 0
            or member.gid != 0
            or member.mtime != 0
            or member.uname != "root"
            or member.gname != "root"
            or member.mode != expected_mode
            or member.linkname
            or member.pax_headers
        ):
            raise VerificationError(
                f"Non-deterministic tar metadata: {expected_name}"
            )
        if expected_name in known_sizes:
            if member.size != known_sizes[expected_name]:
                raise VerificationError(
                    f"Tar member size changed: {expected_name}"
                )
        elif not 0 <= member.size <= SMALL_FILE_LIMIT:
            raise VerificationError(
                f"Unpinned metadata member is too large: {expected_name}"
            )
        if is_directory:
            if member.size != 0:
                raise VerificationError(
                    f"Tar directory contains a payload: {expected_name}"
                )
            continue

        total_payload += member.size
        if total_payload > MAX_PAYLOAD_BYTES:
            raise VerificationError("Tar payload exceeds the allowed total")
        relative = expected_name[len(root) + 1:]
        keep = not relative.startswith("checkpoints/")
        collected = bytearray() if keep else None
        digest = hashlib.sha256()
        remaining = member.size
        while remaining:
            chunk = reader.read_exact(min(4 * 1024 * 1024, remaining))
            remaining -= len(chunk)
            digest.update(chunk)
            if collected is not None:
                collected.extend(chunk)
        padding_size = (-member.size) % tarfile.BLOCKSIZE
        padding = reader.read_exact(padding_size)
        if any(padding):
            raise VerificationError(
                f"Non-zero tar member padding: {expected_name}"
            )
        tar_offset += member.size + padding_size
        identities[relative] = {
            "bytes": member.size,
            "sha256": digest.hexdigest(),
        }
        if collected is not None:
            small_files[relative] = bytes(collected)

    terminal_size = (
        (
            tar_offset
            + (2 * tarfile.BLOCKSIZE)
            + tarfile.RECORDSIZE
            - 1
        )
        // tarfile.RECORDSIZE
        * tarfile.RECORDSIZE
        - tar_offset
    )
    terminal = reader.read_exact(terminal_size)
    if any(terminal):
        raise VerificationError("Non-zero or non-canonical tar EOF padding")
    actual_outer = reader.finish()
    if reader.uncompressed_bytes != tar_offset + terminal_size:
        raise VerificationError("Uncompressed tar length is not canonical")
    return actual_outer, identities, small_files


def _parse_internal_checksums(raw: bytes, expected: set[str]) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise VerificationError("SHA256SUMS is not ASCII") from error
    records: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise VerificationError(f"Malformed SHA256SUMS row: {line!r}")
        name = _safe_relative_name(match.group(2))
        if name in records:
            raise VerificationError(f"Duplicate SHA256SUMS path: {name}")
        records[name] = match.group(1)
    if set(records) != expected:
        raise VerificationError("SHA256SUMS file set changed")
    return records


def _validate_manifest(
    raw: bytes,
    spec: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    expected_source_commit: str,
) -> dict:
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot parse bundle manifest: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "bundle_id", "release_id", "source_repository",
        "release_spec", "identity_audit", "checkpoint_count", "total_checkpoint_bytes",
        "checkpoints", "files", "external_inputs",
    }:
        raise VerificationError("Bundle manifest schema changed")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("bundle_id") != "paper-core-12-primary-a04"
        or manifest.get("release_id") != spec["release_id"]
        or manifest.get("checkpoint_count") != 12
        or manifest.get("total_checkpoint_bytes") != spec["public_total_bytes"]
    ):
        raise VerificationError("Bundle manifest identity changed")
    source = manifest.get("source_repository")
    if (
        not isinstance(source, dict)
        or set(source) != {"url", "commit"}
        or source.get("url") != SOURCE_REPOSITORY
        or source.get("commit") != expected_source_commit
    ):
        raise VerificationError("Invalid source repository identity")
    spec_record = manifest.get("release_spec")
    expected_spec_identity = identities[SPEC_MEMBER]
    if spec_record != {
        "path": SPEC_MEMBER,
        "bytes": expected_spec_identity["bytes"],
        "sha256": expected_spec_identity["sha256"],
    }:
        raise VerificationError("Manifest release-spec identity changed")
    audit_record = manifest.get("identity_audit")
    expected_audit_identity = identities[AUDIT_MEMBER]
    if audit_record != {
        "path": AUDIT_MEMBER,
        "bytes": expected_audit_identity["bytes"],
        "sha256": expected_audit_identity["sha256"],
    }:
        raise VerificationError("Manifest identity-audit record changed")
    audit_receipt = spec["identity_audit"]
    if expected_audit_identity != {
        "bytes": audit_receipt["report_bytes"],
        "sha256": audit_receipt["report_sha256"],
    }:
        raise VerificationError("Bundled identity audit differs from the pinned receipt")
    external = manifest.get("external_inputs")
    if external != {"pretrained_model": spec["external_pretrained_model"]}:
        raise VerificationError("Manifest external-input contract changed")

    checkpoint_records = manifest.get("checkpoints")
    if not isinstance(checkpoint_records, list) or len(checkpoint_records) != 12:
        raise VerificationError("Manifest checkpoint count changed")
    spec_by_experiment = {row["experiment_id"]: row for row in spec["records"]}
    if [row.get("experiment_id") for row in checkpoint_records] != [
        row["experiment_id"] for row in spec["records"]
    ]:
        raise VerificationError("Manifest checkpoint order/scope changed")
    for row in checkpoint_records:
        experiment_id = row["experiment_id"]
        frozen = spec_by_experiment[experiment_id]
        expected = {
            "experiment_id": experiment_id,
            "paper_method": frozen["paper_method"],
            "internal_method": frozen["internal_method"],
            "training_seed": frozen["training_seed"],
            "partition_seed": frozen["partition_seed"],
            "rank": frozen["rank"],
            "selected_round": frozen["selected_round"],
            "path": f"checkpoints/{frozen['public_file_name']}",
            "bytes": frozen["public_identity"]["bytes"],
            "sha256": frozen["public_identity"]["sha256"],
            "tensor_fingerprint": frozen["public_identity"]["tensor_fingerprint"],
            "historical_source": frozen["historical"],
            "path_replacements": frozen["public_identity"]["path_replacements"],
        }
        if row != expected:
            raise VerificationError(f"Manifest checkpoint record changed: {experiment_id}")

    file_records = manifest.get("files")
    expected_payload_names = set(identities) - {
        "BUNDLE_MANIFEST.json", "SHA256SUMS"
    }
    if not isinstance(file_records, list) or len(file_records) != len(expected_payload_names):
        raise VerificationError("Manifest payload-file count changed")
    by_path = {}
    for row in file_records:
        if not isinstance(row, dict) or set(row) != {"path", "role", "bytes", "sha256"}:
            raise VerificationError("Malformed manifest file record")
        name = _safe_relative_name(row.get("path"))
        if name in by_path or name not in expected_payload_names:
            raise VerificationError(f"Unexpected manifest file record: {name}")
        if (
            row["bytes"] != identities[name]["bytes"]
            or row["sha256"] != identities[name]["sha256"]
        ):
            raise VerificationError(f"Manifest file identity mismatch: {name}")
        expected_role = (
            "sanitized_validation_selected_checkpoint"
            if name.startswith("checkpoints/")
            else {
                "README.md": "bundle_readme",
                "LICENSE": "license",
                "THIRD_PARTY_NOTICES.md": "third_party_notices",
                SPEC_MEMBER: "pinned_release_specification",
                AUDIT_MEMBER: "public_identity_discovery_audit",
            }[name]
        )
        if row["role"] != expected_role:
            raise VerificationError(f"Manifest file role changed: {name}")
        by_path[name] = row
    if set(by_path) != expected_payload_names:
        raise VerificationError("Manifest payload-file set changed")
    return manifest


def verify(
    archive: Path,
    checksum: Path,
    spec_path: Path = DEFAULT_SPEC,
    *,
    expected_source_commit: Optional[str] = None,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    spec, spec_raw = load_pinned_spec(spec_path)
    required_commit = (
        source_commit(project_root)
        if expected_source_commit is None
        else expected_source_commit
    )
    if not re.fullmatch(r"[0-9a-f]{40}", required_commit):
        raise VerificationError("Invalid expected source commit")
    committed_spec = _git_blob(
        project_root,
        required_commit,
        "artifacts/paper_core_checkpoint_release_spec.json",
    )
    if spec_raw != committed_spec:
        raise VerificationError(
            "Release specification differs from the required source commit"
        )

    archive_input = archive.expanduser()
    if archive_input.is_symlink():
        raise VerificationError("Archive must not be a symlink")
    try:
        archive_path = archive_input.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"Missing archive: {error}") from error
    if archive_path.name != spec["archive_name"]:
        raise VerificationError("Unexpected archive file")
    expected_outer = _parse_outer_checksum(checksum, archive_path.name)

    expected_relative = _expected_relative_files(spec)
    expected_static = {
        "README.md": expected_bundle_readme(spec, required_commit),
        "LICENSE": expected_license(project_root, required_commit),
        "THIRD_PARTY_NOTICES.md": expected_third_party_notice(
            project_root, required_commit
        ),
    }

    with archive_path.open("rb") as raw_stream:
        before = os.fstat(raw_stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("Archive is not a regular file")
        archive_bytes = before.st_size
        if not 0 < archive_bytes <= MAX_ARCHIVE_BYTES:
            raise VerificationError("Archive size is outside the allowed range")

        actual_outer = _sha256_stream(raw_stream)
        if actual_outer != expected_outer:
            raise VerificationError("Outer archive SHA-256 mismatch")
        raw_stream.seek(0)
        parsed_outer, identities, small_files = _parse_canonical_tar_stream(
            raw_stream,
            spec,
            spec_raw,
            expected_static,
        )
        if parsed_outer != actual_outer:
            raise VerificationError("Archive bytes changed during tar verification")

        raw_stream.seek(0)
        final_outer = _sha256_stream(raw_stream)
        after = os.fstat(raw_stream.fileno())
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
        )
        if any(
            getattr(before, key) != getattr(after, key) for key in stable_fields
        ):
            raise VerificationError("Archive changed during verification")
        if final_outer != actual_outer:
            raise VerificationError("Archive bytes changed during verification")
        path_after = os.stat(archive_path, follow_symlinks=False)
        if (path_after.st_dev, path_after.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise VerificationError("Archive path changed during verification")

    if set(identities) != expected_relative:
        raise VerificationError("Archive file allowlist changed")
    if small_files.get(SPEC_MEMBER) != spec_raw:
        raise VerificationError("Bundled release specification is not byte-identical")

    for name, expected_raw in expected_static.items():
        if small_files.get(name) != expected_raw:
            raise VerificationError(f"Bundled static file differs from source: {name}")

    expected_by_path = {
        f"checkpoints/{row['public_file_name']}": row["public_identity"]
        for row in spec["records"]
    }
    for name, public in expected_by_path.items():
        if identities[name] != {
            "bytes": public["bytes"], "sha256": public["sha256"]
        }:
            raise VerificationError(f"Pinned public checkpoint identity mismatch: {name}")

    checksum_targets = expected_relative - {"SHA256SUMS"}
    checksums = _parse_internal_checksums(
        small_files["SHA256SUMS"], checksum_targets
    )
    for name, digest in checksums.items():
        if identities[name]["sha256"] != digest:
            raise VerificationError(f"Internal SHA-256 mismatch: {name}")
    for name, raw in small_files.items():
        if name.startswith("checkpoints/"):
            continue
        if any(marker in raw for marker in FORBIDDEN_TEXT) or re.search(
            rb"(?<![A-Za-z])[A-Za-z]:[\\/]", raw
        ):
            raise VerificationError(f"Private path marker found in {name}")
    manifest = _validate_manifest(
        small_files["BUNDLE_MANIFEST.json"],
        spec,
        identities,
        required_commit,
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "archive": {
            "file_name": archive_path.name,
            "bytes": archive_bytes,
            "sha256": actual_outer,
        },
        "release_id": spec["release_id"],
        "source_commit": manifest["source_repository"]["commit"],
        "checkpoint_count": 12,
        "public_checkpoint_bytes": spec["public_total_bytes"],
        "release_spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = verify(args.archive, args.checksum, args.spec)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print(_canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
