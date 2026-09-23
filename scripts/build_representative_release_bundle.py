#!/usr/bin/env python3
"""Build the minimal, path-sanitized representative checkpoint release.

The historical checkpoint and schema-v7 split manifest are treated as immutable
inputs.  The builder verifies their frozen identities, derives a compact public
test-replay manifest, removes the two historical pretrained-weight paths from a
trusted copy of the checkpoint, proves that every tensor is unchanged, and then
creates a checksummed release archive.  AOD-4 images and ``rtdetr-l.pt`` are
external inputs and are never bundled.

Two subcommands are intentionally separate:

``manifest``
    Deterministically derive the path-free replay manifest from the historical
    schema-v7 split.  Maintainers use this to update the committed artifact.

``bundle``
    Build the public checkpoint archive on the artifact host.  The committed
    replay manifest must exactly equal a fresh derivation from the historical
    split before the checkpoint is opened.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_checkpoint as evaluation  # noqa: E402


REPLAY_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_ID = "seed42-fedlora-a-r8-a04-replay"
RELEASE_VERSION = "1.0.0"
ARCHIVE_NAME = "fedlora-representative-replay-seed42-v1.0.0.tar.gz"
ARCHIVE_ROOT = "representative-replay"
CHECKPOINT_ASSET_NAME = (
    "seed_42__fl_fedsa_lora_r8_a0.4__best_federated.pt"
)
REPLAY_ASSET_NAME = (
    "seed_42__fl_fedsa_lora_r8_a0.4__replay_manifest.json"
)
COMMITTED_REPLAY = PROJECT_ROOT / "artifacts" / REPLAY_ASSET_NAME
EXPECTED_PATH_FIELDS = {
    ("architecture", "model_weight_path"),
    ("experiment", "model_weights"),
}
PUBLIC_PRETRAINED_PATH = "external/rtdetr-l.pt"
SOURCE_REPOSITORY = (
    "https://github.com/tedrudwls/"
    "Where-to-Adapt-and-What-to-Share-Federated-LoRA-for-Aerial-Object-Detection"
)
DATASET_DOI = "10.17632/cd5z895tr2.1"


class BundleError(RuntimeError):
    """A fail-closed release-construction error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_bytes(raw: bytes, label: str) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"Cannot decode {label}: {error}") from error
    if not isinstance(payload, dict):
        raise BundleError(f"{label} must be a JSON object")
    return payload


def _read_verified(
    path: Path, *, label: str, sha256: str, size: Optional[int] = None
) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BundleError(f"{label} must not be a symlink: {path}")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise BundleError(f"Missing {label}: {path}") from error
    if not resolved.is_file():
        raise BundleError(f"{label} must be a regular non-symlink file: {path}")
    if size is not None and resolved.stat().st_size != size:
        raise BundleError(
            f"{label} byte-size mismatch: expected={size}, "
            f"actual={resolved.stat().st_size}"
        )
    raw = resolved.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise BundleError(
            f"{label} SHA-256 mismatch: expected={sha256}, actual={actual}"
        )
    return raw


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be an object")
    return value


def build_replay_manifest(split: Mapping[str, Any]) -> dict:
    """Return the only public split metadata required by the P0 evaluator."""

    metadata = _require_mapping(split.get("metadata"), "split.metadata")
    clients = split.get("clients")
    if not isinstance(clients, list) or len(clients) != 3:
        raise BundleError("Historical split must contain exactly three clients")

    mismatches = {
        key: {"historical": metadata.get(key), "required": value}
        for key, value in evaluation.FROZEN_MANIFEST_PROTOCOL.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise BundleError(
            "Historical split protocol changed: "
            + json.dumps(mismatches, sort_keys=True)
        )

    split_counts = metadata.get("split_counts")
    source_split_counts = metadata.get("source_split_counts")
    if split_counts != source_split_counts:
        raise BundleError("Historical source/split aggregate counts disagree")
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "val", "test"}:
        raise BundleError("Historical split counts are incomplete")

    ordered_clients = sorted(clients, key=lambda row: int(row.get("client_id", -1)))
    if [int(row.get("client_id", -1)) for row in ordered_clients] != [0, 1, 2]:
        raise BundleError("Historical split client IDs must be 0,1,2")
    client_counts: dict[str, list[int]] = {
        split_name: [] for split_name in ("train", "val", "test")
    }
    client_test_ids = []
    for client_id, client in enumerate(ordered_clients):
        blocks = _require_mapping(client.get("splits"), f"client {client_id} splits")
        for split_name in ("train", "val", "test"):
            block = _require_mapping(
                blocks.get(split_name), f"client {client_id} {split_name}"
            )
            count = int(block.get("num_images", -1))
            image_ids = [int(value) for value in block.get("image_ids", [])]
            if count <= 0 or len(image_ids) != count or len(image_ids) != len(set(image_ids)):
                raise BundleError(
                    f"Client {client_id} {split_name} assignment is invalid"
                )
            client_counts[split_name].append(count)
            if split_name == "test":
                client_test_ids.append({
                    "client_id": client_id,
                    "image_ids": image_ids,
                })
        if any(
            sum(client_counts[name]) > int(split_counts[name]["images"])
            for name in ("train", "val", "test")
        ):
            raise BundleError("Client assignments exceed aggregate split counts")
    for split_name, counts in client_counts.items():
        if sum(counts) != int(split_counts[split_name]["images"]):
            raise BundleError(f"Client {split_name} counts do not cover the split")

    inventory = _require_mapping(
        metadata.get("source_hash_inventory"), "source hash inventory"
    )
    if int(inventory.get("schema_version", -1)) != 1:
        raise BundleError("Historical source inventory schema changed")
    records_block = _require_mapping(inventory.get("records"), "inventory records")
    tree_block = _require_mapping(
        inventory.get("per_split_image_tree_sha256"), "inventory tree digests"
    )
    test_records = records_block.get("test")
    if not isinstance(test_records, list):
        raise BundleError("Historical split has no test image inventory")
    normalized_records = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for number, row in enumerate(test_records):
        if not isinstance(row, list) or len(row) != 3:
            raise BundleError(f"Malformed test inventory row {number}")
        image_id = int(row[0])
        relative_name = evaluation._safe_relative_name(row[1])
        digest = str(row[2])
        if image_id in seen_ids or relative_name in seen_names or not evaluation._is_sha256(digest):
            raise BundleError(f"Duplicate/invalid test inventory row {number}")
        seen_ids.add(image_id)
        seen_names.add(relative_name)
        normalized_records.append([image_id, relative_name, digest])
    expected_tree = str(tree_block.get("test", ""))
    if evaluation._inventory_tree_digest(normalized_records) != expected_tree:
        raise BundleError("Historical test inventory tree digest is inconsistent")
    flattened_test = [
        image_id
        for client in client_test_ids
        for image_id in client["image_ids"]
    ]
    if len(flattened_test) != len(set(flattened_test)) or set(flattened_test) != seen_ids:
        raise BundleError("Client test assignments are not a disjoint inventory cover")

    annotation_sha = str(
        _require_mapping(metadata.get("annotation_sha256"), "annotation hashes").get(
            "test", ""
        )
    )
    if not evaluation._is_sha256(annotation_sha):
        raise BundleError("Historical test annotation SHA-256 is invalid")
    category_audit = _require_mapping(
        _require_mapping(metadata.get("source_category_audit"), "category audit").get(
            "test"
        ),
        "test category audit",
    )

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "protocol": "fedlora_representative_test_replay",
        "experiment_id": evaluation.TARGET_EXPERIMENT_ID,
        "historical_split_manifest": {
            "schema_version": int(metadata["schema_version"]),
            "sha256": evaluation.FROZEN_SPLIT_SHA256,
        },
        "partition_protocol": copy.deepcopy(evaluation.FROZEN_MANIFEST_PROTOCOL),
        "split_counts": copy.deepcopy(split_counts),
        "client_image_counts": client_counts,
        "test": {
            "annotation": {
                "relative_path": "test/_annotations.coco.json",
                "sha256": annotation_sha,
            },
            "category_audit": copy.deepcopy(category_audit),
            "source_inventory": {
                "schema_version": 1,
                "identity_policy": inventory.get("identity_policy"),
                "historical_inventory_sha256": inventory.get("inventory_sha256"),
                "image_tree_sha256": expected_tree,
                "records": normalized_records,
            },
            "client_image_ids": client_test_ids,
        },
    }


def _path_string(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("file://")
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )


def _walk_strings(value: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield path + (f"<dict-key:{key}>",), key
            yield from _walk_strings(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, path + (index,))
    elif isinstance(value, (set, frozenset)):
        for index, item in enumerate(sorted(value, key=repr)):
            yield from _walk_strings(item, path + (f"<set-item:{index}>",))
    elif isinstance(value, str):
        yield path, value


def sanitize_checkpoint_payload(payload: Any) -> tuple[dict, list[dict]]:
    if not isinstance(payload, dict):
        raise BundleError("Historical checkpoint payload is not a dictionary")
    observed = {
        path: value for path, value in _walk_strings(payload) if _path_string(value)
    }
    if set(observed) != EXPECTED_PATH_FIELDS:
        formatted = {"/" + "/".join(map(str, key)): value for key, value in observed.items()}
        raise BundleError(
            "Checkpoint absolute-path inventory differs from the reviewed contract: "
            + json.dumps(formatted, sort_keys=True)
        )
    for path, value in observed.items():
        if PurePosixPath(value).name != "rtdetr-l.pt":
            raise BundleError(f"Unexpected checkpoint path at {path}: {value!r}")

    sanitized = copy.deepcopy(payload)
    replacements = []
    for first, second in sorted(EXPECTED_PATH_FIELDS):
        old = sanitized[first][second]
        sanitized[first][second] = PUBLIC_PRETRAINED_PATH
        replacements.append({
            "json_pointer": f"/{first}/{second}",
            "historical_value_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
            "public_value": PUBLIC_PRETRAINED_PATH,
        })
    remaining = {
        path: value for path, value in _walk_strings(sanitized) if _path_string(value)
    }
    if remaining:
        raise BundleError("Sanitized checkpoint still contains an absolute path")
    return sanitized, replacements


def _tensor_fingerprint(payload: Any, torch: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        nonlocal count
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps(list(path), separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            if tensor.numel():
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            digest.update(b"\0")
            count += 1
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], path + (key,))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, path + (index,))

    visit(payload, ())
    if count == 0:
        raise BundleError("Checkpoint contains no tensor state")
    return digest.hexdigest(), count


def _assert_payload_equivalent(
    historical: Any,
    public: Any,
    torch: Any,
    path: tuple[Any, ...] = (),
) -> None:
    if path in EXPECTED_PATH_FIELDS:
        if not isinstance(historical, str) or public != PUBLIC_PRETRAINED_PATH:
            raise BundleError(f"Invalid sanitized value at {path}")
        return
    if torch.is_tensor(historical):
        if not torch.is_tensor(public) or historical.dtype != public.dtype:
            raise BundleError(f"Tensor type changed at {path}")
        if tuple(historical.shape) != tuple(public.shape) or not torch.equal(
            historical.detach().cpu(), public.detach().cpu()
        ):
            raise BundleError(f"Tensor value changed at {path}")
        return
    if type(historical) is not type(public):
        raise BundleError(f"Payload type changed at {path}")
    if isinstance(historical, dict):
        if set(historical) != set(public):
            raise BundleError(f"Payload keys changed at {path}")
        for key in historical:
            _assert_payload_equivalent(
                historical[key], public[key], torch, path + (key,)
            )
    elif isinstance(historical, (list, tuple)):
        if len(historical) != len(public):
            raise BundleError(f"Payload length changed at {path}")
        for index, (left, right) in enumerate(zip(historical, public)):
            _assert_payload_equivalent(left, right, torch, path + (index,))
    elif historical != public:
        raise BundleError(f"Payload value changed at {path}")


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
        raise BundleError("Bundle must be built from a clean tracked Git checkout") from error
    if status:
        raise BundleError("Bundle must be built from a completely clean Git checkout")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BundleError("Could not resolve an immutable Git commit")
    return commit


def _write_bytes_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(mode)
    except OSError as error:
        raise BundleError(f"Cannot create {path}: {error}") from error


def _archive_tree(source: Path, archive: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in [source, *sorted(source.rglob("*"), key=lambda item: item.as_posix())]:
            relative = path.relative_to(source.parent).as_posix()
            info = tarfile.TarInfo(relative)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                tar.addfile(info)
            elif path.is_file() and not path.is_symlink():
                info.type = tarfile.REGTYPE
                info.mode = 0o644
                info.size = path.stat().st_size
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
            else:
                raise BundleError(f"Unsupported bundle filesystem entry: {path}")
    with archive.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            zipped.write(buffer.getvalue())
        output.flush()
        os.fsync(output.fileno())


def _bundle_readme(commit: str) -> bytes:
    return (
        "# FedLoRA representative replay bundle\n\n"
        "This bundle contains one path-sanitized, validation-selected personalized "
        "federated checkpoint and the compact manifest needed to replay its test AP.\n\n"
        "Not included: AOD-4 images and the Ultralytics RT-DETR-L pretrained weight. "
        f"Obtain AOD-4 from https://doi.org/{DATASET_DOI} and provide `rtdetr-l.pt` "
        f"with SHA-256 `{evaluation.FROZEN_PRETRAINED_SHA256}`.\n\n"
        f"Source repository: {SOURCE_REPOSITORY}\n"
        f"Source commit: `{commit}`\n\n"
        "Verify the outer `.sha256` file before extracting, then verify `SHA256SUMS`. "
        "Use the read-only command documented in `docs/READ_ONLY_EVALUATION.md` at "
        "the recorded source commit. Never load a checkpoint whose published hash "
        "has not first been verified.\n"
    ).encode("utf-8")


def _bundle_third_party_notice(project_root: Path, commit: str) -> bytes:
    source = project_root / "THIRD_PARTY_NOTICES.md"
    text = source.read_text(encoding="utf-8")
    relative_link = "[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)"
    immutable_url = (
        f"[`docs/CHECKPOINTS.md`]({SOURCE_REPOSITORY}/blob/{commit}/"
        "docs/CHECKPOINTS.md)"
    )
    if text.count(relative_link) != 1:
        raise BundleError("Third-party notice checkpoint link changed unexpectedly")
    return text.replace(relative_link, immutable_url).encode("utf-8")


def _build_bundle(args: argparse.Namespace) -> None:
    project_root = args.project_dir.expanduser().resolve(strict=True)
    if project_root != PROJECT_ROOT.resolve():
        raise BundleError(
            f"This builder must run from its own checkout: expected={PROJECT_ROOT}, "
            f"received={project_root}"
        )
    commit = _git_commit(project_root)
    split_raw = _read_verified(
        args.split_file,
        label="historical split manifest",
        sha256=evaluation.FROZEN_SPLIT_SHA256,
    )
    derived_replay_raw = _canonical_json_bytes(
        build_replay_manifest(_load_json_bytes(split_raw, "historical split manifest"))
    )
    committed_replay_raw = COMMITTED_REPLAY.read_bytes()
    if committed_replay_raw != derived_replay_raw:
        raise BundleError(
            "Committed replay manifest differs from the historical split derivation"
        )
    checkpoint_input = args.checkpoint.expanduser()
    checkpoint_raw = _read_verified(
        checkpoint_input,
        label="historical checkpoint",
        sha256=evaluation.FROZEN_CHECKPOINT_SHA256,
        size=evaluation.FROZEN_CHECKPOINT_BYTES,
    )
    checkpoint = checkpoint_input.resolve(strict=True)
    source_before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        for path in (checkpoint, args.split_file.resolve(strict=True), COMMITTED_REPLAY)
    }

    try:
        import torch
    except ImportError as error:
        raise BundleError("PyTorch is required to sanitize the checkpoint") from error

    historical_payload = torch.load(
        io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True
    )
    historical_path_values = [
        str(historical_payload[first][second])
        for first, second in sorted(EXPECTED_PATH_FIELDS)
    ]
    public_payload, replacements = sanitize_checkpoint_payload(historical_payload)
    historical_tensor_sha, tensor_count = _tensor_fingerprint(historical_payload, torch)
    public_tensor_sha, public_tensor_count = _tensor_fingerprint(public_payload, torch)
    if (historical_tensor_sha, tensor_count) != (public_tensor_sha, public_tensor_count):
        raise BundleError("Tensor fingerprint changed during checkpoint sanitization")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.name != ARCHIVE_ROOT:
        raise BundleError(f"Output directory must end with {ARCHIVE_ROOT!r}")
    publication_dir = output_dir.parent
    publication_parent = publication_dir.parent
    if publication_dir.exists():
        raise BundleError(
            f"Atomic publication parent already exists: {publication_dir}"
        )
    publication_parent.mkdir(parents=True, exist_ok=True)
    archive = publication_dir / ARCHIVE_NAME
    checksum_file = publication_dir / f"{ARCHIVE_NAME}.sha256"

    with tempfile.TemporaryDirectory(
        prefix="fedlora-public-bundle-", dir=publication_parent
    ) as temporary:
        temporary_root = Path(temporary)
        staged_publication = temporary_root / publication_dir.name
        staged_publication.mkdir(mode=0o755)
        stage = staged_publication / ARCHIVE_ROOT
        stage.mkdir(mode=0o755)
        public_checkpoint = stage / "checkpoint" / CHECKPOINT_ASSET_NAME
        public_checkpoint.parent.mkdir(mode=0o755)
        with public_checkpoint.open("xb") as stream:
            torch.save(public_payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        public_checkpoint.chmod(0o644)
        reloaded = torch.load(
            str(public_checkpoint), map_location="cpu", weights_only=True
        )
        _assert_payload_equivalent(historical_payload, reloaded, torch)
        reloaded_tensor_sha, reloaded_tensor_count = _tensor_fingerprint(reloaded, torch)
        if (reloaded_tensor_sha, reloaded_tensor_count) != (
            historical_tensor_sha,
            tensor_count,
        ):
            raise BundleError("Saved public checkpoint tensor fingerprint changed")
        if any(_path_string(value) for _, value in _walk_strings(reloaded)):
            raise BundleError("Saved public checkpoint contains an absolute path")
        public_checkpoint_raw = public_checkpoint.read_bytes()
        residual_markers = [
            value for value in historical_path_values
            if value.encode("utf-8") in public_checkpoint_raw
        ]
        forbidden_markers = [
            marker.decode("ascii")
            for marker in (b"/home/", b"/Users/", b"gpuadmin", b"file://")
            if marker in public_checkpoint_raw
        ]
        if residual_markers or forbidden_markers or re.search(
            rb"(?<![A-Za-z])[A-Za-z]:[\\/]", public_checkpoint_raw
        ):
            raise BundleError(
                "Serialized public checkpoint retains a private path marker: "
                + json.dumps(
                    {
                        "reviewed_paths": residual_markers,
                        "forbidden_markers": forbidden_markers,
                        "windows_absolute_path": bool(re.search(
                            rb"(?<![A-Za-z])[A-Za-z]:[\\/]",
                            public_checkpoint_raw,
                        )),
                    },
                    sort_keys=True,
                )
            )

        replay_path = stage / "metadata" / REPLAY_ASSET_NAME
        _write_bytes_exclusive(replay_path, committed_replay_raw)
        _write_bytes_exclusive(
            stage / "LICENSE", (project_root / "LICENSE").read_bytes()
        )
        _write_bytes_exclusive(
            stage / "THIRD_PARTY_NOTICES.md",
            _bundle_third_party_notice(project_root, commit),
        )
        _write_bytes_exclusive(stage / "README.md", _bundle_readme(commit))

        described = [
            (public_checkpoint, "sanitized_validation_selected_checkpoint"),
            (replay_path, "path_free_test_replay_manifest"),
            (stage / "LICENSE", "license"),
            (stage / "THIRD_PARTY_NOTICES.md", "third_party_notices"),
            (stage / "README.md", "bundle_readme"),
        ]
        files = [
            {
                "path": path.relative_to(stage).as_posix(),
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, role in described
        ]
        files[0]["historical_source"] = {
            "bytes": evaluation.FROZEN_CHECKPOINT_BYTES,
            "sha256": evaluation.FROZEN_CHECKPOINT_SHA256,
        }
        files[0]["path_replacements"] = replacements
        files[0]["tensor_fingerprint"] = {
            "algorithm": "recursive_path_dtype_shape_raw_bytes_sha256_v1",
            "tensor_count": tensor_count,
            "sha256": historical_tensor_sha,
        }
        bundle_manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": BUNDLE_ID,
            "release_version": RELEASE_VERSION,
            "experiment_id": evaluation.TARGET_EXPERIMENT_ID,
            "method": "FedLoRA-A (Share-A / local B)",
            "source_repository": {"url": SOURCE_REPOSITORY, "commit": commit},
            "files": files,
            "historical_split_manifest_sha256": evaluation.FROZEN_SPLIT_SHA256,
            "external_inputs": {
                "pretrained_model": {
                    "included": False,
                    "file_name": "rtdetr-l.pt",
                    "sha256": evaluation.FROZEN_PRETRAINED_SHA256,
                },
                "dataset": {
                    "included": False,
                    "doi": DATASET_DOI,
                    "test_images": 2241,
                    "test_image_tree_sha256": (
                        build_replay_manifest(
                            _load_json_bytes(split_raw, "historical split manifest")
                        )["test"]["source_inventory"]["image_tree_sha256"]
                    ),
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
        manifest_path = stage / "BUNDLE_MANIFEST.json"
        _write_bytes_exclusive(manifest_path, _canonical_json_bytes(bundle_manifest))
        checksum_targets = sorted(
            [*described, (manifest_path, "bundle_manifest")],
            key=lambda item: item[0].relative_to(stage).as_posix(),
        )
        checksum_text = "".join(
            f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
            for path, _ in checksum_targets
        )
        _write_bytes_exclusive(stage / "SHA256SUMS", checksum_text.encode("utf-8"))

        staged_archive = staged_publication / ARCHIVE_NAME
        staged_checksum = staged_publication / f"{ARCHIVE_NAME}.sha256"
        _archive_tree(stage, staged_archive)
        archive_sha = sha256_file(staged_archive)
        _write_bytes_exclusive(
            staged_checksum,
            f"{archive_sha}  {ARCHIVE_NAME}\n".encode("ascii"),
        )
        public_checkpoint_sha = sha256_file(public_checkpoint)
        for raw_path, expected in source_before.items():
            path = Path(raw_path)
            actual = (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
            if actual != expected:
                raise BundleError(
                    f"Protected input changed during bundle creation: {path}"
                )
        if sha256_file(staged_archive) != archive_sha:
            raise BundleError("Staged archive changed before publication")
        if sha256_file(public_checkpoint) != public_checkpoint_sha:
            raise BundleError("Staged checkpoint changed before publication")
        # All content gates have passed. One same-filesystem directory rename
        # publishes the extracted tree, archive and outer checksum together.
        os.replace(staged_publication, publication_dir)

    if sha256_file(archive) != archive_sha:
        raise BundleError("Published archive changed during atomic promotion")
    if sha256_file(output_dir / "checkpoint" / CHECKPOINT_ASSET_NAME) != public_checkpoint_sha:
        raise BundleError("Published checkpoint changed during atomic promotion")
    print(f"[PASS] representative public bundle: {archive}")
    print(f"archive_sha256={archive_sha}")
    print(f"public_checkpoint_sha256={public_checkpoint_sha}")
    print(f"tensor_fingerprint_sha256={historical_tensor_sha}")
    print(f"replay_manifest_sha256={hashlib.sha256(committed_replay_raw).hexdigest()}")
    print(f"bundle_directory={output_dir}")
    print(f"outer_checksum={checksum_file}")


def _emit_manifest(args: argparse.Namespace) -> None:
    raw = _read_verified(
        args.split_file,
        label="historical split manifest",
        sha256=evaluation.FROZEN_SPLIT_SHA256,
    )
    public = _canonical_json_bytes(
        build_replay_manifest(_load_json_bytes(raw, "historical split manifest"))
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise BundleError(f"Output already exists: {output}")
    _write_bytes_exclusive(output, public)
    print(f"[PASS] replay manifest: {output}")
    print(f"bytes={len(public)}")
    print(f"sha256={hashlib.sha256(public).hexdigest()}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest", help="derive the public replay manifest")
    manifest.add_argument("--split-file", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_emit_manifest)
    bundle = subparsers.add_parser("bundle", help="build the sanitized release archive")
    bundle.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    bundle.add_argument("--checkpoint", type=Path, required=True)
    bundle.add_argument("--split-file", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.set_defaults(handler=_build_bundle)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        args.handler(args)
        return 0
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
