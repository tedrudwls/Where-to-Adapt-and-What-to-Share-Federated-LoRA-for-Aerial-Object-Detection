#!/usr/bin/env python3
"""Build the audited 12-checkpoint paper-core release archive.

Each historical checkpoint is verified, restricted-loaded, contract-checked,
path-sanitized, serialized and matched against an independently pinned public
identity. Only one checkpoint is resident at a time. The deterministic tar.gz
is written in streaming mode, structurally verified, and then published with a
same-filesystem atomic directory rename. Historical inputs are never edited.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_paper_core_release_inputs as audit  # noqa: E402
from scripts import build_representative_release_bundle as representative  # noqa: E402
from scripts import verify_paper_core_release_bundle as verifier  # noqa: E402
from scripts.verify_checkpoint_assets import load_index  # noqa: E402


SOURCE_REPOSITORY = verifier.SOURCE_REPOSITORY
SPEC_MEMBER = verifier.SPEC_MEMBER
AUDIT_MEMBER = verifier.AUDIT_MEMBER
LICENSE_NAME = "LICENSE"
NOTICE_NAME = "THIRD_PARTY_NOTICES.md"
README_NAME = "README.md"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"


class BundleError(RuntimeError):
    """A fail-closed paper-core release build error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _git_commit(project_root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BundleError("Build must use a clean tracked Git checkout") from error
    if status:
        raise BundleError("Build checkout is not completely clean")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BundleError("Cannot resolve an immutable source commit")
    return commit


def _write_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(mode)
    except OSError as error:
        raise BundleError(f"Cannot create output file: {error}") from error


def _source_snapshot(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, sha256_file(path)


def _safe_source_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise BundleError(f"Unsafe source path in release spec: {relative!r}")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise BundleError(f"Historical checkpoint must not be a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise BundleError(f"Invalid historical checkpoint path: {relative}") from error
    if not resolved.is_file():
        raise BundleError(f"Historical checkpoint is not a regular file: {relative}")
    return resolved


def _runtime(torch: Any) -> dict[str, str]:
    return {"python": platform.python_version(), "torch": str(torch.__version__)}


def _identity(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _bundle_readme(spec: Mapping[str, Any], commit: str) -> bytes:
    return verifier.expected_bundle_readme(spec, commit)


def _third_party_notice(project_root: Path, commit: str) -> bytes:
    try:
        return verifier.expected_third_party_notice(project_root, commit)
    except Exception as error:
        raise BundleError(f"Cannot prepare third-party notice: {error}") from error


def _stream_identity_and_private_marker_gate(
    path: Path, historical_path_values: Sequence[str]
) -> dict[str, Any]:
    reviewed = [value.encode("utf-8") for value in historical_path_values]
    markers = [*reviewed, *representative.RAW_FORBIDDEN_MARKERS]
    max_marker = max((len(marker) for marker in markers), default=1)
    digest = hashlib.sha256()
    total = 0
    carry = b""
    found = set()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
            window = carry + chunk
            for marker in markers:
                if marker in window:
                    found.add(marker)
            carry = window[-(max_marker - 1):] if max_marker > 1 else b""
    if found:
        residual_paths = [
            value
            for value in historical_path_values
            if value.encode("utf-8") in found
        ]
        forbidden = [
            marker.decode("ascii")
            for marker in representative.RAW_FORBIDDEN_MARKERS
            if marker in found
        ]
        raise BundleError(
            "Serialized public checkpoint retains a private path marker: "
            + json.dumps(
                {
                    "reviewed_paths": residual_paths,
                    "forbidden_markers": forbidden,
                },
                sort_keys=True,
            )
        )
    return {"bytes": total, "sha256": digest.hexdigest()}


def _role(name: str) -> str:
    if name.startswith("checkpoints/"):
        return "sanitized_validation_selected_checkpoint"
    return {
        README_NAME: "bundle_readme",
        LICENSE_NAME: "license",
        NOTICE_NAME: "third_party_notices",
        SPEC_MEMBER: "pinned_release_specification",
        AUDIT_MEMBER: "public_identity_discovery_audit",
    }[name]


def _build_manifest(
    spec: Mapping[str, Any],
    commit: str,
    payload_identities: Mapping[str, Mapping[str, Any]],
) -> dict:
    checkpoints = []
    for record in spec["records"]:
        public = record["public_identity"]
        checkpoints.append({
            "experiment_id": record["experiment_id"],
            "paper_method": record["paper_method"],
            "internal_method": record["internal_method"],
            "training_seed": record["training_seed"],
            "partition_seed": record["partition_seed"],
            "rank": record["rank"],
            "selected_round": record["selected_round"],
            "path": f"checkpoints/{record['public_file_name']}",
            "bytes": public["bytes"],
            "sha256": public["sha256"],
            "tensor_fingerprint": public["tensor_fingerprint"],
            "historical_source": record["historical"],
            "path_replacements": public["path_replacements"],
        })
    files = [
        {
            "path": name,
            "role": _role(name),
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
        }
        for name, identity in sorted(payload_identities.items())
    ]
    spec_identity = payload_identities[SPEC_MEMBER]
    return {
        "schema_version": 1,
        "bundle_id": "paper-core-12-primary-a04",
        "release_id": spec["release_id"],
        "source_repository": {"url": SOURCE_REPOSITORY, "commit": commit},
        "release_spec": {
            "path": SPEC_MEMBER,
            "bytes": spec_identity["bytes"],
            "sha256": spec_identity["sha256"],
        },
        "identity_audit": {
            "path": AUDIT_MEMBER,
            "bytes": payload_identities[AUDIT_MEMBER]["bytes"],
            "sha256": payload_identities[AUDIT_MEMBER]["sha256"],
        },
        "checkpoint_count": 12,
        "total_checkpoint_bytes": spec["public_total_bytes"],
        "checkpoints": checkpoints,
        "files": files,
        "external_inputs": {
            "pretrained_model": spec["external_pretrained_model"]
        },
    }


def _tar_info(name: str, *, size: int = 0, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.mode = 0o755 if directory else 0o644
    info.size = 0 if directory else size
    if directory:
        info.type = tarfile.DIRTYPE
    else:
        info.type = tarfile.REGTYPE
    return info


def _add_bytes(tar: tarfile.TarFile, name: str, raw: bytes) -> None:
    tar.addfile(_tar_info(name, size=len(raw)), io.BytesIO(raw))


def _add_file(tar: tarfile.TarFile, name: str, path: Path) -> None:
    with path.open("rb") as stream:
        tar.addfile(_tar_info(name, size=path.stat().st_size), stream)


def _verify_public_candidate(
    source: Path,
    index_record: Mapping[str, Any],
    spec_record: Mapping[str, Any],
    torch: Any,
    temporary_root: Path,
) -> Path:
    expected_historical = spec_record["historical"]
    actual_historical = _file_identity(source)
    if actual_historical != {
        "bytes": expected_historical["bytes"],
        "sha256": expected_historical["sha256"],
    }:
        raise BundleError(
            f"Historical checkpoint identity mismatch: {spec_record['experiment_id']}"
        )
    try:
        payload = torch.load(str(source), map_location="cpu", weights_only=True)
    except Exception as error:
        raise BundleError(
            f"Restricted load failed for {spec_record['experiment_id']}: "
            f"{type(error).__name__}: {error}"
        ) from error
    audit.validate_checkpoint_contract(payload, index_record, torch)
    historical_paths = [
        value
        for _, value in representative._walk_strings(payload)
        if representative._path_string(value)
    ]
    public_payload, replacements = representative.sanitize_checkpoint_payload(payload)
    historical_tensor_sha, historical_tensor_count = representative._tensor_fingerprint(
        payload, torch
    )
    public_tensor_sha, public_tensor_count = representative._tensor_fingerprint(
        public_payload, torch
    )
    if (historical_tensor_sha, historical_tensor_count) != (
        public_tensor_sha, public_tensor_count
    ):
        raise BundleError("Tensor fingerprint changed during path sanitization")

    target = temporary_root / spec_record["public_file_name"]
    with target.open("xb") as stream:
        torch.save(public_payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o600)
    del public_payload
    gc.collect()
    try:
        reloaded = torch.load(str(target), map_location="cpu", weights_only=True)
    except Exception as error:
        raise BundleError(
            f"Restricted public-candidate reload failed: {type(error).__name__}: {error}"
        ) from error
    representative._assert_payload_equivalent(payload, reloaded, torch)
    reloaded_sha, reloaded_count = representative._tensor_fingerprint(reloaded, torch)
    if (reloaded_sha, reloaded_count) != (
        historical_tensor_sha, historical_tensor_count
    ):
        raise BundleError("Serialized public checkpoint changed tensor state")
    if any(
        representative._path_string(value)
        for _, value in representative._walk_strings(reloaded)
    ):
        raise BundleError("Serialized public checkpoint retains an absolute path")
    serialized = _stream_identity_and_private_marker_gate(target, historical_paths)
    observed = {
        "bytes": serialized["bytes"],
        "sha256": serialized["sha256"],
        "tensor_fingerprint": {
            "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
            "tensor_count": reloaded_count,
            "sha256": reloaded_sha,
        },
        "path_replacements": replacements,
        "residual_absolute_path_count": 0,
    }
    if observed != spec_record["public_identity"]:
        raise BundleError(
            "Public checkpoint differs from the independently pinned identity: "
            f"{spec_record['experiment_id']}"
        )
    del reloaded, payload
    gc.collect()
    return target


def build(args: argparse.Namespace) -> dict:
    project_root = args.project_dir.expanduser().resolve(strict=True)
    if project_root != PROJECT_ROOT.resolve():
        raise BundleError("Builder must run from its own clean checkout")
    source_root = args.source_project.expanduser().resolve(strict=True)
    commit = _git_commit(project_root)
    spec_input = args.spec.expanduser()
    index_input = args.index.expanduser()
    audit_input = args.audit_report.expanduser()
    if spec_input.is_symlink() or index_input.is_symlink() or audit_input.is_symlink():
        raise BundleError("Release metadata inputs must not be symlinks")
    spec_path = spec_input.resolve(strict=True)
    index_path = index_input.resolve(strict=True)
    audit_report_path = audit_input.resolve(strict=True)
    spec, spec_raw = verifier.load_pinned_spec(spec_path)
    try:
        selected = audit.select_core_records(load_index(index_path))
    except Exception as error:
        raise BundleError(f"Cannot load checkpoint index: {error}") from error
    index_identity = {
        "relative_path": "artifacts/checkpoint_index.json",
        "bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
    }
    try:
        audit._validate_spec_against_index(spec, selected, index_identity)
    except Exception as error:
        raise BundleError(f"Release spec/index gate failed: {error}") from error

    try:
        import torch
    except ImportError as error:
        raise BundleError("PyTorch is required to build checkpoint assets") from error
    if _runtime(torch) != spec["serialization_runtime"]:
        raise BundleError(
            "Serialization runtime differs from the audited identity runtime: "
            f"actual={_runtime(torch)}, required={spec['serialization_runtime']}"
        )

    spec_by_experiment = {row["experiment_id"]: row for row in spec["records"]}
    sources = []
    for record in selected:
        spec_record = spec_by_experiment[record["experiment_id"]]
        source = _safe_source_file(
            source_root, spec_record["historical"]["project_relative_path"]
        )
        sources.append((source, record, spec_record))
    protected = {
        path: _source_snapshot(path)
        for path in [
            index_path,
            spec_path,
            audit_report_path,
            *[row[0] for row in sources],
        ]
    }

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise BundleError(f"Output directory already exists: {output_dir}")
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    archive_name = spec["archive_name"]
    checksum_name = f"{archive_name}.sha256"
    root = spec["archive_root"]

    license_raw = verifier.expected_license(project_root, commit)
    notice_raw = _third_party_notice(project_root, commit)
    readme_raw = _bundle_readme(spec, commit)
    audit_raw = audit_report_path.read_bytes()
    audit_receipt = spec["identity_audit"]
    if _identity(audit_raw) != {
        "bytes": audit_receipt["report_bytes"],
        "sha256": audit_receipt["report_sha256"],
    }:
        raise BundleError("Committed identity audit differs from the pinned receipt")
    static_payloads = {
        README_NAME: readme_raw,
        LICENSE_NAME: license_raw,
        NOTICE_NAME: notice_raw,
        SPEC_MEMBER: spec_raw,
        AUDIT_MEMBER: audit_raw,
    }
    payload_identities = {
        name: _identity(raw) for name, raw in static_payloads.items()
    }
    for record in spec["records"]:
        public = record["public_identity"]
        payload_identities[f"checkpoints/{record['public_file_name']}"] = {
            "bytes": public["bytes"], "sha256": public["sha256"]
        }
    manifest = _build_manifest(spec, commit, payload_identities)
    manifest_raw = _canonical_json_bytes(manifest)
    checksummed = {
        **payload_identities,
        MANIFEST_NAME: _identity(manifest_raw),
    }
    sums_raw = "".join(
        f"{identity['sha256']}  {name}\n"
        for name, identity in sorted(checksummed.items())
    ).encode("ascii")

    with tempfile.TemporaryDirectory(
        prefix="fedlora-paper-core-build-", dir=output_parent
    ) as temporary:
        temporary_root = Path(temporary)
        staged_output = temporary_root / output_dir.name
        staged_output.mkdir(mode=0o700)
        staged_archive = staged_output / archive_name
        checkpoint_temporary = temporary_root / "checkpoint"
        checkpoint_temporary.mkdir(mode=0o700)
        with staged_archive.open("xb") as raw_archive:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_archive, mtime=0, compresslevel=9
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
                ) as tar:
                    for directory in (
                        root, f"{root}/checkpoints", f"{root}/metadata"
                    ):
                        tar.addfile(_tar_info(directory, directory=True))
                    _add_bytes(tar, f"{root}/{MANIFEST_NAME}", manifest_raw)
                    _add_bytes(tar, f"{root}/{CHECKSUMS_NAME}", sums_raw)
                    for name in (
                        README_NAME,
                        LICENSE_NAME,
                        NOTICE_NAME,
                        SPEC_MEMBER,
                        AUDIT_MEMBER,
                    ):
                        _add_bytes(tar, f"{root}/{name}", static_payloads[name])
                    for source, index_record, spec_record in sources:
                        candidate = _verify_public_candidate(
                            source,
                            index_record,
                            spec_record,
                            torch,
                            checkpoint_temporary,
                        )
                        _add_file(
                            tar,
                            f"{root}/checkpoints/{spec_record['public_file_name']}",
                            candidate,
                        )
                        candidate.unlink()
            raw_archive.flush()
            os.fsync(raw_archive.fileno())
        staged_archive.chmod(0o600)
        archive_sha = sha256_file(staged_archive)
        staged_checksum = staged_output / checksum_name
        _write_exclusive(
            staged_checksum,
            f"{archive_sha}  {archive_name}\n".encode("ascii"),
            mode=0o600,
        )
        verification = verifier.verify(
            staged_archive,
            staged_checksum,
            spec_path,
            expected_source_commit=commit,
            project_root=project_root,
        )
        _write_exclusive(
            staged_output / "build_verification.json",
            _canonical_json_bytes(verification),
            mode=0o600,
        )
        for path, before in protected.items():
            if _source_snapshot(path) != before:
                raise BundleError(f"Protected input changed during build: {path.name}")
        staged_descriptor = os.open(str(staged_output), os.O_RDONLY)
        try:
            os.fsync(staged_descriptor)
        finally:
            os.close(staged_descriptor)
        os.replace(staged_output, output_dir)
        parent_descriptor = os.open(str(output_parent), os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    final_archive = output_dir / archive_name
    final_checksum = output_dir / checksum_name
    final_report = verifier.verify(
        final_archive,
        final_checksum,
        spec_path,
        expected_source_commit=commit,
        project_root=project_root,
    )
    if final_report != verification:
        raise BundleError("Verification changed after atomic publication")
    return final_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint_index.json",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "paper_core_checkpoint_release_spec.json",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "paper_core_input_audit.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        report = build(_parser().parse_args(argv))
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("[PASS] 12-checkpoint paper-core public bundle built and verified")
    print(_canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
