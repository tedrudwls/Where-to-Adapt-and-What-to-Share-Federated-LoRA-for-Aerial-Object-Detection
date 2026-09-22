#!/usr/bin/env python3
"""Read-only robustness audit for the primary FL membership-inference results.

This program deliberately does not call ``main.py --resume`` or
``evaluate_federated_checkpoint`` because those evaluation paths rewrite the
primary ``fl_results.json``.  It restores only the validation-selected
``best_federated.pt`` bundles, extracts per-image losses into a separate audit
directory, and verifies that every primary result/checkpoint is byte-identical
when the process exits.

The first version focuses on the two controls that are most important for the
AOD-4 protocol:

* repeated, source-group-atomic attack calibration/evaluation splits; and
* a fresh four-class target-initialization control and matched delta loss
  (trained loss minus initial loss) for diagnosing train/test split-origin
  confounding.

Every persisted image identifier, source identifier and content identifier is
SHA-256 hashed.  Raw filenames are used only in memory to locate generated
YOLO symlinks and are never written to the audit output.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence


sys.dont_write_bytecode = True
AUDIT_SCHEMA_VERSION = 1
RECORD_CACHE_SCHEMA_VERSION = 1
SUPPORTED_METHODS = ("full_ft", "lora", "fedsa_lora", "fixed_share_b_lora")
SUPPORTED_AUDIT_SEEDS = (42, 43, 44)
EXPERIMENT_NAMES = {
    "full_ft": "fl_full_ft_a0.4",
    "lora": "fl_lora_r8_a0.4",
    "fedsa_lora": "fl_fedsa_lora_r8_a0.4",
    "fixed_share_b_lora": "fl_fixed_share_b_lora_r8_a0.4",
}
SCORE_FIELDS = ("trained_loss", "initial_loss", "delta_loss")
FPR_TARGETS = (0.01, 0.05, 0.10)
EXPECTED_AUGMENTATION_PROTOCOL = {
    "implementation": "ultralytics_8.4.126_RTDETRDataset",
    "initial": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
        "fliplr": 0.5, "bgr": 0.0, "mosaic": 1.0,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "copy_paste_mode": "flip",
    },
    "close_mosaic_effective_epochs_requested": 10,
    "close_mosaic_disables": ["mosaic", "mixup", "cutmix", "copy_paste"],
    "rect": False,
    "cache": False,
    "train_only": True,
    "persistent_dataloader_workers": False,
}
_ACTIVE_PLAN_PATH: Path | None = None
_ACTIVE_RUN_NONCE: str | None = None


def audit_output_name(seed: int) -> str:
    seed = int(seed)
    if seed not in SUPPORTED_AUDIT_SEEDS:
        raise ValueError(
            f"security_audit_v1 supports seeds {SUPPORTED_AUDIT_SEEDS}; received {seed}"
        )
    return "security_audit_v1" if seed == 42 else f"security_audit_v1_seed{seed}"


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload, path: os.PathLike | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)


def atomic_text_dump(text: str, path: os.PathLike | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)


def load_json(path: os.PathLike | str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _path_record(path: os.PathLike | str) -> dict:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(f"Integrity input is not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def snapshot_files(paths: Iterable[os.PathLike | str]) -> dict:
    resolved = sorted({str(Path(path).resolve(strict=True)) for path in paths})
    return {path: _path_record(path) for path in resolved}


def snapshot_files_best_effort(paths: Iterable[os.PathLike | str]) -> dict:
    """Snapshot failure-path inputs while representing deletion instead of raising."""
    output = {}
    for raw_path in sorted({str(path) for path in paths}):
        path = Path(raw_path)
        try:
            resolved = str(path.resolve(strict=True))
            output[resolved] = _path_record(resolved)
        except (FileNotFoundError, OSError) as error:
            output[str(path.resolve(strict=False))] = {
                "path": str(path.resolve(strict=False)),
                "missing_or_unreadable": True,
                "error": f"{type(error).__name__}: {error}",
            }
    return output


def changed_snapshot(before: Mapping[str, dict], after: Mapping[str, dict]) -> list[dict]:
    changes = []
    for path in sorted(set(before) | set(after)):
        left, right = before.get(path), after.get(path)
        if left != right:
            changes.append({"path": path, "before": left, "after": right})
    return changes


def validate_output_location(results_root: os.PathLike | str,
                             output_dir: os.PathLike | str,
                             primary_dirs: Sequence[os.PathLike | str],
                             seed: int = 42) -> Path:
    root = Path(results_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    expected = root / audit_output_name(seed)
    if output != expected:
        raise ValueError(
            "The read-only audit writes only to "
            f"{expected}; received {output}"
        )
    for primary in primary_dirs:
        primary_path = Path(primary).resolve(strict=True)
        if output == primary_path or primary_path in output.parents:
            raise ValueError(f"Audit output overlaps a primary experiment: {primary_path}")
    return output


def validate_resume_state(output_dir: os.PathLike | str, *, resume: bool,
                          dry_run: bool) -> None:
    """Permit only a first run or a genuinely interrupted exact-cache resume."""
    output = Path(output_dir)
    if output.exists() and not resume:
        raise FileExistsError(
            f"Audit output already exists: {output}. Use --resume only to reuse "
            "strictly validated caches, or preserve it and create a new audit version."
        )
    if (
        output.exists()
        and resume
        and not dry_run
        and (output / "audit_report.json").exists()
    ):
        raise FileExistsError(
            "A final audit_report.json already exists. Completed or ambiguous audit "
            f"directories are immutable and cannot be resumed: {output}"
        )


def validate_existing_resume_plan(output_dir: os.PathLike | str, new_plan: Mapping,
                                  *, resume: bool, dry_run: bool) -> None:
    """Reject protocol/input drift before replacing an interrupted run plan."""
    output = Path(output_dir)
    if not output.exists() or not resume or dry_run:
        return
    plan_path = output / "audit_plan.json"
    if not plan_path.exists():
        # The shell wrapper creates the private directory and console log before
        # the first Python invocation, so a missing plan is the valid first-run case.
        return
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"Unsafe interrupted audit plan: {plan_path}")
    previous = load_json(plan_path)
    comparable_keys = (
        "audit_schema_version", "mode", "read_only_primary", "seed",
        "audit_instance", "methods", "output_dir", "runtime_cache_root",
        "protected_inputs", "protocol",
    )
    mismatches = {
        key: {"previous": previous.get(key), "current": new_plan.get(key)}
        for key in comparable_keys
        if previous.get(key) != new_plan.get(key)
    }
    if mismatches:
        raise ValueError(
            "Interrupted audit plan/protected-input mismatch; refusing resume: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )
    integrity_path = output / "integrity_manifest.json"
    if integrity_path.exists():
        if integrity_path.is_symlink() or not integrity_path.is_file():
            raise ValueError(f"Unsafe interrupted integrity manifest: {integrity_path}")
        integrity = load_json(integrity_path)
        if integrity.get("read_only_primary_gate") is not True:
            raise ValueError(
                "Interrupted audit recorded a protected-primary integrity failure; "
                "refusing resume"
            )


def resolve_primary_inputs(results_root: os.PathLike | str, seed: int,
                           methods: Sequence[str]) -> dict:
    root = Path(results_root).resolve(strict=True)
    resolved = {}
    for method in methods:
        if method not in EXPERIMENT_NAMES:
            raise ValueError(f"Unsupported method: {method}")
        experiment_dir = root / f"seed_{int(seed)}" / EXPERIMENT_NAMES[method]
        result_path = experiment_dir / "fl_results.json"
        best_path = experiment_dir / "weights" / "best_federated.pt"
        last_path = experiment_dir / "weights" / "last_federated.pt"
        missing = [str(path) for path in (result_path, best_path, last_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing primary artifact(s): " + ", ".join(missing))
        resolved[method] = {
            "experiment_dir": str(experiment_dir.resolve()),
            "result": str(result_path.resolve()),
            "best_checkpoint": str(best_path.resolve()),
            "last_checkpoint": str(last_path.resolve()),
        }
    return resolved


def _strict_primary_result(payload: dict, *, method: str, seed: int,
                           split_sha256: str, best_checkpoint: str) -> None:
    checks = {
        "status": (payload.get("status"), "complete"),
        "mode": (payload.get("mode"), "fl"),
        "fl_method": (payload.get("fl_method"), method),
        "seed": (int(payload.get("seed", -1)), int(seed)),
        "partition_seed": (int(payload.get("partition_seed", -1)), int(seed)),
        "partition": (payload.get("partition"), "dirichlet"),
        "num_clients": (int(payload.get("num_clients", -1)), 3),
        "rounds_executed": (int(payload.get("rounds_executed", -1)), 20),
        "local_epochs": (int(payload.get("local_epochs", -1)), 5),
        "split_manifest_sha256": (
            str(payload.get("split_manifest_sha256", "")).lower(), split_sha256
        ),
    }
    if not math.isclose(float(payload.get("dirichlet_alpha", float("nan"))), 0.4,
                        rel_tol=0.0, abs_tol=1e-12):
        checks["dirichlet_alpha"] = (payload.get("dirichlet_alpha"), 0.4)
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{method}: result has no selection block")
    checks.update({
        "selection.criterion": (
            selection.get("criterion"), "best_macro_client_local_val_AP"
        ),
        "selection.checkpoint.basename": (
            os.path.basename(str(selection.get("checkpoint", ""))),
            os.path.basename(best_checkpoint),
        ),
    })
    mismatches = [
        f"{name}: primary={left!r}, expected={right!r}"
        for name, (left, right) in checks.items() if left != right
    ]
    if mismatches:
        raise ValueError(f"{method}: primary result mismatch: " + "; ".join(mismatches))


def _validate_architecture_provenance(*, method: str,
                                      checkpoint_architecture: dict,
                                      primary_architecture: dict,
                                      model_weight_sha256: str) -> str:
    """Bind a primary result to the checkpoint's complete architecture manifest."""
    if not isinstance(checkpoint_architecture, dict) or not isinstance(
        primary_architecture, dict
    ):
        raise ValueError(f"{method}: missing checkpoint/primary architecture metadata")
    checkpoint_digest = json_sha256(checkpoint_architecture)
    primary_digest = json_sha256(primary_architecture)
    if checkpoint_digest != primary_digest:
        raise ValueError(
            f"{method}: checkpoint/primary architecture manifests differ: "
            f"checkpoint_sha256={checkpoint_digest}, primary_sha256={primary_digest}"
        )
    observed_weight_sha = checkpoint_architecture.get("model_weight_sha256")
    if observed_weight_sha != model_weight_sha256:
        raise ValueError(
            f"{method}: architecture model-weight SHA-256 mismatch: "
            f"observed={observed_weight_sha!r}, expected={model_weight_sha256!r}"
        )
    initial_digest = checkpoint_architecture.get(
        "initial_target_tensor_state_sha256"
    )
    if not isinstance(initial_digest, str) or len(initial_digest) != 64:
        raise ValueError(f"{method}: checkpoint has no valid initial target digest")
    return initial_digest


def _validate_primary_checkpoint_compatibility(compatibility: Mapping, *,
                                               method: str, seed: int) -> str:
    """Enforce the exact primary RT-DETR training configuration before auditing."""
    uses_lora = method != "full_ft"
    expected = {
        "fl_method": method,
        "model_name": "rtdetr-l",
        "num_classes": 4,
        "num_clients": 3,
        "fl_rounds": 20,
        "local_epochs": 5,
        "batch_size": 8,
        "img_size": 640,
        "num_workers": 4,
        "lr": 0.0003 if uses_lora else 0.0001,
        "head_lr": 0.0001,
        "backbone_lr_ratio": 0.1,
        "weight_decay": 0.0001,
        "warmup_epochs": 5.0,
        "min_lr_ratio": 0.01,
        "grad_clip_norm": 0.1,
        "close_mosaic_epochs": 10,
        "fedprox_mu": 0.0,
        "reset_optimizer_each_round": True,
        "amp": False,
        "augmentation_protocol": EXPECTED_AUGMENTATION_PROTOCOL,
        "seed": int(seed),
        "partition_seed": int(seed),
        "partition": "dirichlet",
        "dirichlet_alpha": 0.4,
        "lora_rank": 8 if uses_lora else None,
        "lora_alpha": 16.0 if uses_lora else None,
        "lora_dropout": 0.0 if uses_lora else None,
        "apply_lora_backbone": True if uses_lora else None,
        "apply_lora_decoder": True if uses_lora else None,
        "backbone_min_channels": 64 if uses_lora else None,
    }
    mismatches = {
        key: {"checkpoint": compatibility.get(key), "expected": value}
        for key, value in expected.items()
        if compatibility.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{method}: checkpoint is not the frozen primary configuration: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )
    normalized = {
        key: compatibility[key]
        for key in expected
        if key not in ("seed", "partition_seed")
    }
    return json_sha256(normalized)


def _args_from_checkpoint_compatibility(compatibility: dict, *, data_root: str,
                                        split_file: str, model_weights: str,
                                        device: str) -> SimpleNamespace:
    required = {
        "fl_method", "model_name", "num_classes", "num_clients", "fl_rounds",
        "local_epochs", "batch_size", "img_size", "num_workers", "lr", "head_lr",
        "backbone_lr_ratio", "weight_decay", "warmup_epochs", "min_lr_ratio",
        "grad_clip_norm", "close_mosaic_epochs", "fedprox_mu",
        "reset_optimizer_each_round", "amp", "seed", "partition_seed",
        "partition", "dirichlet_alpha", "lora_rank", "lora_alpha", "lora_dropout",
        "apply_lora_backbone", "apply_lora_decoder", "backbone_min_channels",
    }
    missing = sorted(required - set(compatibility))
    if missing:
        raise ValueError(f"Checkpoint compatibility manifest is incomplete: {missing}")
    values = dict(compatibility)
    values.update({
        "data_root": os.path.abspath(data_root),
        "split_file": os.path.abspath(split_file),
        "model_weights": os.path.abspath(model_weights),
        "device": device,
        "rehash_source_images": False,
        "min_bbox_area": 0.0,
        "min_bbox_side": 0.0,
        "run_mia": False,
        "cross_client_eval": False,
        "visualize_interval": 0,
        "vis_samples": 0,
        "mia_max_samples": 1000,
        "mia_calibration_fraction": 0.5,
        "patience": 0,
        "val_interval": 5,
    })
    return SimpleNamespace(**values)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _minimal_data_info(split_info: dict, split_file: str) -> dict:
    metadata = split_info.get("metadata")
    if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != 7:
        raise ValueError("The audit requires an immutable schema-7 split manifest")
    clients = sorted(split_info.get("clients", []), key=lambda row: int(row["client_id"]))
    num_clients = int(metadata.get("num_clients", -1))
    if [int(row["client_id"]) for row in clients] != list(range(num_clients)):
        raise ValueError("Split client IDs are incomplete or non-contiguous")
    split_path = Path(split_file).resolve(strict=True)
    split_stem = split_path.stem
    yolo_dir = split_path.parent / split_stem.replace("split_", "yolo_", 1)
    client_yamls = [str(yolo_dir / f"client_{client_id}" / "dataset.yaml")
                    for client_id in range(num_clients)]
    full_yaml = str(yolo_dir / "full" / "dataset.yaml")
    for path in [*client_yamls, full_yaml]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Generated dataset YAML is missing: {path}")
    return {
        "client_yamls": client_yamls,
        "client_sizes": [
            int(client["splits"]["train"]["num_images"]) for client in clients
        ],
        "client_eval_sizes": [
            {
                split: int(client["splits"][split]["num_images"])
                for split in ("val", "test")
            }
            for client in clients
        ],
        "full_yaml": full_yaml,
        "class_names": list(metadata["class_names"]),
        "client_splits": [
            [int(value) for value in client["splits"]["train"]["image_ids"]]
            for client in clients
        ],
        "train_coco": {},
        "split_manifest_sha256": sha256_file(split_path),
        "split_metadata": metadata,
        "yolo_dir": str(yolo_dir),
    }


def _load_catalog(split_info: dict, data_root: str) -> tuple[dict, dict]:
    """Reconstruct audited split metadata without calling the cache-writing prepare_data."""
    from data.dataset import (
        SOURCE_GROUP_FIELD,
        apply_source_split_policy,
        build_image_ann_map,
        canonicalize_aod4_categories,
        category_mapping,
        filter_annotations,
        load_coco_annotations,
        validate_source_hash_inventory,
    )

    metadata = split_info["metadata"]
    annotation_hashes = metadata.get("annotation_sha256", {})
    coco_by_split = {}
    for split in ("train", "val", "test"):
        path = Path(data_root) / split / "_annotations.coco.json"
        if sha256_file(path) != annotation_hashes.get(split):
            raise ValueError(f"{split}: annotation JSON differs from the split manifest")
        raw = load_coco_annotations(str(path))
        canonical, _ = canonicalize_aod4_categories(raw)
        coco_by_split[split] = canonical
    inventory = metadata.get("source_hash_inventory")
    inventory_maps = validate_source_hash_inventory(coco_by_split, inventory)
    selected, current_audit = apply_source_split_policy(
        coco_by_split, inventory, policy=metadata["source_split_policy"]
    )
    if current_audit != metadata.get("cross_split_source_audit"):
        raise ValueError("Reconstructed source audit differs from split manifest")
    selected = {
        split: filter_annotations(
            selected[split], float(metadata["min_bbox_area"]),
            float(metadata["min_bbox_side"]), drop_empty_images=False,
        )
        for split in ("train", "val", "test")
    }
    category_names = category_mapping(selected["train"])
    catalogs = {}
    for split in ("train", "test"):
        annotations = build_image_ann_map(selected[split])
        rows = {}
        for image in selected[split]["images"]:
            image_id = int(image["id"])
            image_annotations = [ann for ann in annotations.get(image_id, [])
                                 if not ann.get("iscrowd", 0)]
            class_counts = defaultdict(int)
            areas = []
            width, height = int(image["width"]), int(image["height"])
            for annotation in image_annotations:
                class_counts[category_names[int(annotation["category_id"])]] += 1
                _, _, box_width, box_height = map(float, annotation["bbox"])
                areas.append(max(0.0, box_width) * max(0.0, box_height) / (width * height))
            content_sha = inventory_maps[split][image_id]["sha256"]
            source_group = str(image[SOURCE_GROUP_FIELD])
            public = {
                "sample_id_sha256": hashlib.sha256(
                    f"{split}\0{image_id}\0{content_sha}".encode("utf-8")
                ).hexdigest(),
                "image_content_sha256": content_sha,
                "source_group_sha256": hashlib.sha256(
                    source_group.encode("utf-8")
                ).hexdigest(),
                "split": split,
                "width": width,
                "height": height,
                "object_count": len(image_annotations),
                "background": not image_annotations,
                "class_counts": dict(sorted(class_counts.items())),
                "classes": sorted(class_counts),
                "bbox_area_ratio_mean": float(sum(areas) / len(areas)) if areas else 0.0,
                "bbox_area_ratio_min": float(min(areas)) if areas else 0.0,
                "bbox_area_ratio_max": float(max(areas)) if areas else 0.0,
            }
            rows[image_id] = {
                **public,
                "_image_id": image_id,
                "_file_name": str(image["file_name"]).replace("\\", "/"),
            }
        catalogs[split] = rows
    return catalogs, selected


def _candidate_sets(split_info: dict, catalogs: dict, selected: dict,
                    *, selection_seed: int, max_member: int,
                    max_local_nonmember: int, max_pooled_nonmember: int) -> dict:
    from data.dataset import SOURCE_GROUP_FIELD

    clients = sorted(split_info["clients"], key=lambda row: int(row["client_id"]))
    global_train_groups = {
        str(image[SOURCE_GROUP_FIELD]) for image in selected["train"]["images"]
    }
    test_group_by_id = {
        int(image["id"]): str(image[SOURCE_GROUP_FIELD])
        for image in selected["test"]["images"]
    }

    def select(rows: Sequence[dict], maximum: int, label: str) -> list[dict]:
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{selection_seed}\0{label}\0{row['sample_id_sha256']}".encode("ascii")
            ).hexdigest(),
        )
        return ordered[:min(int(maximum), len(ordered))]

    eligible_pooled_ids = [
        image_id for image_id in catalogs["test"]
        if test_group_by_id[image_id] not in global_train_groups
    ]
    pooled = select(
        [catalogs["test"][image_id] for image_id in eligible_pooled_ids],
        max_pooled_nonmember, "pooled_test",
    )
    result = {}
    for client in clients:
        client_id = int(client["client_id"])
        train_ids = [int(value) for value in client["splits"]["train"]["image_ids"]]
        test_ids = [int(value) for value in client["splits"]["test"]["image_ids"]]
        eligible_test = [
            image_id for image_id in test_ids
            if test_group_by_id[image_id] not in global_train_groups
        ]
        result[client_id] = {
            "member": select(
                [catalogs["train"][image_id] for image_id in train_ids],
                max_member, f"client_{client_id}_train",
            ),
            "local_nonmember": select(
                [catalogs["test"][image_id] for image_id in eligible_test],
                max_local_nonmember, f"client_{client_id}_local_test",
            ),
            "pooled_nonmember": list(pooled),
        }
    return result


@contextlib.contextmanager
def _forbid_ultralytics_cache_writes():
    """Fail closed instead of allowing RTDETRDataset to refresh label caches."""
    import ultralytics.data.dataset as dataset_module

    original = dataset_module.save_dataset_cache_file

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Read-only audit refused an Ultralytics label-cache write. Run the normal "
            "dataset preflight first so validated caches exist; do not grant write access."
        )

    dataset_module.save_dataset_cache_file = forbidden
    try:
        yield
    finally:
        dataset_module.save_dataset_cache_file = original


def _dataset_relative_name(path: str, split: str) -> str:
    normalized = str(path).replace("\\", "/")
    marker = f"/{split}/images/"
    if marker not in normalized:
        raise ValueError(f"Cannot derive generated relative filename from {path}")
    return normalized.rsplit(marker, 1)[1].lstrip("/")


def _extract_losses(model, data_yaml: str, split: str, args,
                    candidates: Sequence[dict]) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from trainers.trainer import (
        _move_batch,
        _seed_worker,
        _unpack_loss,
        build_rtdetr_dataset,
    )

    with _forbid_ultralytics_cache_writes():
        dataset = build_rtdetr_dataset(data_yaml, split, args, augment=False)
    index_by_name = {}
    for index, image_path in enumerate(dataset.im_files):
        relative = _dataset_relative_name(image_path, split)
        if relative in index_by_name:
            raise ValueError(f"Duplicate generated dataset filename: {relative}")
        index_by_name[relative] = index
    missing = [row["_file_name"] for row in candidates if row["_file_name"] not in index_by_name]
    if missing:
        raise ValueError(f"Candidate images are missing from {data_yaml}: {missing[:5]}")
    indices = [index_by_name[row["_file_name"]] for row in candidates]
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + (31 if split == "train" else 47))
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=max(0, int(args.num_workers)), pin_memory=str(args.device).startswith("cuda"),
        collate_fn=dataset.collate_fn, worker_init_fn=_seed_worker,
        generator=generator, persistent_workers=False, drop_last=False,
    )
    device = torch.device(args.device)
    original_device = next(model.model.parameters()).device
    was_training = model.model.training
    model.model.to(device)
    model.model.eval()
    model.enforce_frozen_norm_eval()
    output = []
    try:
        with torch.no_grad():
            for candidate, batch in zip(candidates, loader):
                batch_data = _move_batch(batch, device)
                loss, components = _unpack_loss(model.model(batch_data))
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite loss for {candidate['sample_id_sha256']}")
                record = {key: value for key, value in candidate.items()
                          if not key.startswith("_")}
                record["loss"] = value
                record["loss_components"] = {
                    str(key): float(component.detach().cpu())
                    for key, component in components.items()
                    if hasattr(component, "detach") and component.numel() == 1
                }
                output.append(record)
    finally:
        model.model.train(was_training)
        model.enforce_frozen_norm_eval()
        model.model.to(original_device)
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if len(output) != len(candidates):
        raise RuntimeError(f"Expected {len(candidates)} losses, extracted {len(output)}")
    return output


def _cache_load_or_extract(cache_path: Path, cache_key: dict, *, resume: bool,
                           extractor) -> list[dict]:
    if cache_path.is_symlink():
        raise ValueError(f"Audit cache must not be a symbolic link: {cache_path}")
    if cache_path.exists():
        # main() has already refused a pre-existing audit directory unless the
        # caller supplied --resume.  During a fresh invocation, shared initial
        # caches are legitimately encountered again by the second method.
        cached = load_json(cache_path)
        records = cached.get("records") if isinstance(cached, dict) else None
        valid_envelope = (
            isinstance(cached, dict)
            and int(cached.get("schema_version", -1)) == RECORD_CACHE_SCHEMA_VERSION
            and cached.get("cache_key") == cache_key
            and isinstance(records, list)
            and json_sha256(records) == cached.get("records_sha256")
        )
        valid_records = False
        if valid_envelope:
            ids = [row.get("sample_id_sha256") for row in records
                   if isinstance(row, dict)]
            forbidden = {"file_name", "image_path", "image_id", "_file_name", "_image_id"}
            valid_records = (
                len(ids) == len(records)
                and len(ids) == int(cache_key.get("sample_count", -1))
                and len(ids) == len(set(ids))
                and json_sha256(ids) == cache_key.get("sample_ids_sha256")
                and all(
                    not (forbidden & set(row))
                    and math.isfinite(float(row.get("loss", float("nan"))))
                    for row in records
                )
            )
        if valid_envelope and valid_records:
            print(f"[CACHE] {cache_path}")
            return records
        raise ValueError(
            f"Invalid or stale audit cache: {cache_path}. Preserve the old audit and "
            "start a new versioned audit rather than overwriting it."
        )
    records = extractor()
    ids = [row.get("sample_id_sha256") for row in records if isinstance(row, dict)]
    forbidden = {"file_name", "image_path", "image_id", "_file_name", "_image_id"}
    if not (
        len(ids) == len(records) == int(cache_key.get("sample_count", -1))
        and len(ids) == len(set(ids))
        and json_sha256(ids) == cache_key.get("sample_ids_sha256")
        and all(
            not (forbidden & set(row))
            and math.isfinite(float(row.get("loss", float("nan"))))
            for row in records
        )
    ):
        raise ValueError("Newly extracted loss records failed the strict cache gate")
    payload = {
        "schema_version": RECORD_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "records_sha256": json_sha256(records),
        "records": records,
    }
    atomic_json_dump(payload, cache_path)
    return records


def _merge_losses(trained: Sequence[dict], initial: Sequence[dict]) -> list[dict]:
    initial_by_id = {row["sample_id_sha256"]: row for row in initial}
    if len(initial_by_id) != len(initial):
        raise ValueError("Initial-loss cache contains duplicate sample IDs")
    if {row["sample_id_sha256"] for row in trained} != set(initial_by_id):
        raise ValueError("Trained and initial losses are not paired on identical samples")
    merged = []
    for row in trained:
        sample_id = row["sample_id_sha256"]
        baseline = initial_by_id[sample_id]
        for field in ("source_group_sha256", "split", "class_counts", "object_count"):
            if row[field] != baseline[field]:
                raise ValueError(f"Paired metadata mismatch for {sample_id}: {field}")
        trained_loss, initial_loss = float(row["loss"]), float(baseline["loss"])
        merged.append({
            **{key: value for key, value in row.items() if key not in ("loss", "loss_components")},
            "trained_loss": trained_loss,
            "initial_loss": initial_loss,
            "delta_loss": trained_loss - initial_loss,
            "trained_loss_components": row.get("loss_components", {}),
            "initial_loss_components": baseline.get("loss_components", {}),
        })
    return merged


def _validate_record_pool(members: Sequence[dict], nonmembers: Sequence[dict],
                          score_field: str) -> None:
    if min(len(members), len(nonmembers)) < 4:
        raise ValueError("Each membership class needs at least four records")
    for label, rows in (("member", members), ("nonmember", nonmembers)):
        ids = [row["sample_id_sha256"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {label} sample IDs")
        if any(not math.isfinite(float(row[score_field])) for row in rows):
            raise FloatingPointError(f"Non-finite {label} {score_field}")
    member_groups = {row["source_group_sha256"] for row in members}
    nonmember_groups = {row["source_group_sha256"] for row in nonmembers}
    overlap = member_groups & nonmember_groups
    if overlap:
        raise ValueError(f"Member/nonmember source overlap: {len(overlap)} groups")


def _group_atomic_split(rows: Sequence[dict], fraction: float, rng) -> tuple[list, list]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group_sha256"]].append(row)
    group_rows = list(groups.values())
    if len(group_rows) < 2:
        raise ValueError("Source-group-atomic attack split needs at least two source groups")
    rng.shuffle(group_rows)
    cumulative = 0
    target = len(rows) * float(fraction)
    candidates = []
    for cutoff in range(1, len(group_rows)):
        cumulative += len(group_rows[cutoff - 1])
        candidates.append((abs(cumulative - target), cutoff))
    cutoff = min(candidates)[1]
    calibration = [row for group in group_rows[:cutoff] for row in group]
    evaluation = [row for group in group_rows[cutoff:] for row in group]
    return calibration, evaluation


def _balance(left: Sequence[dict], right: Sequence[dict], rng) -> tuple[list, list]:
    size = min(len(left), len(right))
    if size < 2:
        raise ValueError("Balanced attack partition has fewer than two samples per class")
    left_indices = rng.permutation(len(left))[:size]
    right_indices = rng.permutation(len(right))[:size]
    return ([left[int(index)] for index in left_indices],
            [right[int(index)] for index in right_indices])


def _calibrate_operating_threshold(labels, scores, maximum_fpr: float) -> dict:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    admissible = np.flatnonzero(fpr <= maximum_fpr + 1e-12)
    if not admissible.size:
        raise RuntimeError("Calibration ROC has no admissible low-FPR point")
    best_tpr = float(np.max(tpr[admissible]))
    candidates = [
        int(index) for index in admissible
        if math.isclose(float(tpr[index]), best_tpr, rel_tol=0.0, abs_tol=1e-12)
    ]
    # Prefer the smallest achieved calibration FPR at tied TPR, then the most
    # conservative (largest) finite threshold.  sklearn represents the
    # no-positive rule with +inf; convert it to the next finite float so the
    # audit remains strict JSON (allow_nan=False).
    finite_candidates = [value for value in candidates if np.isfinite(thresholds[value])]
    index = min(finite_candidates or candidates, key=lambda value: (
        float(fpr[value]), -float(thresholds[value])
    ))
    threshold = float(thresholds[index])
    if not math.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
        if not math.isfinite(threshold):
            raise FloatingPointError("Could not construct a finite no-positive threshold")
    return {
        "target_fpr": float(maximum_fpr),
        "threshold": threshold,
        "calibration_tpr": float(tpr[index]),
        "calibration_achieved_fpr": float(fpr[index]),
    }


def _apply_operating_threshold(labels, scores, calibration: dict) -> dict:
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= float(calibration["threshold"])
    member = labels == 1
    nonmember = labels == 0
    true_positives = int(np.sum(predictions & member))
    false_positives = int(np.sum(predictions & nonmember))
    member_count = int(np.sum(member))
    nonmember_count = int(np.sum(np.asarray(labels) == 0))
    return {
        **calibration,
        "tpr": float(true_positives / member_count),
        "achieved_fpr": float(false_positives / nonmember_count),
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "member_count": member_count,
        "nonmember_count": nonmember_count,
    }


def _attack_once(members: Sequence[dict], nonmembers: Sequence[dict], *,
                 score_field: str, calibration_fraction: float, seed: int) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(int(seed))
    cal_m, eval_m = _group_atomic_split(members, calibration_fraction, rng)
    cal_n, eval_n = _group_atomic_split(nonmembers, calibration_fraction, rng)
    cal_m, cal_n = _balance(cal_m, cal_n, rng)
    eval_m, eval_n = _balance(eval_m, eval_n, rng)

    calibration_labels = np.concatenate((np.ones(len(cal_m)), np.zeros(len(cal_n))))
    calibration_raw = np.asarray(
        [-float(row[score_field]) for row in cal_m]
        + [-float(row[score_field]) for row in cal_n], dtype=np.float64,
    )
    raw_auc = float(roc_auc_score(calibration_labels, calibration_raw))
    direction = 1.0 if raw_auc >= 0.5 else -1.0
    calibration_scores = direction * calibration_raw
    fpr, tpr, thresholds = roc_curve(
        calibration_labels, calibration_scores, drop_intermediate=False
    )
    objective = tpr - fpr
    best = np.flatnonzero(np.isclose(objective, np.max(objective), rtol=0, atol=1e-12))
    finite = [int(index) for index in best if np.isfinite(thresholds[index])]
    threshold = float(thresholds[finite[0] if finite else int(best[0])])

    evaluation_labels = np.concatenate((np.ones(len(eval_m)), np.zeros(len(eval_n))))
    evaluation_scores = direction * np.asarray(
        [-float(row[score_field]) for row in eval_m]
        + [-float(row[score_field]) for row in eval_n], dtype=np.float64,
    )
    predictions = (evaluation_scores >= threshold).astype(np.int64)
    operating_points = {
        f"{int(target * 100)}pct": _apply_operating_threshold(
            evaluation_labels,
            evaluation_scores,
            _calibrate_operating_threshold(
                calibration_labels, calibration_scores, target
            ),
        )
        for target in FPR_TARGETS
    }
    split_plan = {}
    for name, rows in (
        ("calibration_members", cal_m), ("calibration_nonmembers", cal_n),
        ("evaluation_members", eval_m), ("evaluation_nonmembers", eval_n),
    ):
        split_plan[name] = {
            "sample_ids_sha256": json_sha256(sorted(
                row["sample_id_sha256"] for row in rows
            )),
            "source_group_ids_sha256": json_sha256(sorted(
                {row["source_group_sha256"] for row in rows}
            )),
            "sample_count": len(rows),
            "source_group_count": len({row["source_group_sha256"] for row in rows}),
        }
    return {
        "seed": int(seed),
        "direction": (
            "lower_score_field_is_more_likely_member"
            if direction > 0 else "higher_score_field_is_more_likely_member"
        ),
        "threshold": threshold,
        "calibration_raw_negative_score_auc": raw_auc,
        "calibration_member_count": len(cal_m),
        "calibration_nonmember_count": len(cal_n),
        "evaluation_member_count": len(eval_m),
        "evaluation_nonmember_count": len(eval_n),
        "calibration_source_groups": {
            "member": len({row["source_group_sha256"] for row in cal_m}),
            "nonmember": len({row["source_group_sha256"] for row in cal_n}),
        },
        "evaluation_source_groups": {
            "member": len({row["source_group_sha256"] for row in eval_m}),
            "nonmember": len({row["source_group_sha256"] for row in eval_n}),
        },
        "split_plan": split_plan,
        "split_plan_sha256": json_sha256(split_plan),
        "auc_roc": float(roc_auc_score(evaluation_labels, evaluation_scores)),
        "asr": float(np.mean(predictions == evaluation_labels)),
        "operating_points": operating_points,
    }


def repeated_attack(members: Sequence[dict], nonmembers: Sequence[dict], *,
                    score_field: str, repeats: int, attack_seed: int,
                    calibration_fraction: float) -> dict:
    _validate_record_pool(members, nonmembers, score_field)
    runs = [
        _attack_once(
            members, nonmembers, score_field=score_field,
            calibration_fraction=calibration_fraction,
            seed=int(attack_seed) + repeat,
        )
        for repeat in range(int(repeats))
    ]
    paths = {
        "auc_roc": lambda row: row["auc_roc"],
        "asr": lambda row: row["asr"],
        "tpr_at_calibration_target_1fpr_threshold": (
            lambda row: row["operating_points"]["1pct"]["tpr"]
        ),
        "evaluation_fpr_at_calibration_target_1fpr_threshold": (
            lambda row: row["operating_points"]["1pct"]["achieved_fpr"]
        ),
        "tpr_at_calibration_target_5fpr_threshold": (
            lambda row: row["operating_points"]["5pct"]["tpr"]
        ),
        "evaluation_fpr_at_calibration_target_5fpr_threshold": (
            lambda row: row["operating_points"]["5pct"]["achieved_fpr"]
        ),
        "tpr_at_calibration_target_10fpr_threshold": (
            lambda row: row["operating_points"]["10pct"]["tpr"]
        ),
        "evaluation_fpr_at_calibration_target_10fpr_threshold": (
            lambda row: row["operating_points"]["10pct"]["achieved_fpr"]
        ),
    }
    summary = {}
    for name, getter in paths.items():
        values = [float(getter(row)) for row in runs]
        summary[name] = {
            "mean": float(statistics.fmean(values)),
            "repeat_sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "median": float(statistics.median(values)),
        }
    return {
        "attack": "source_group_atomic_repeated_calibration_evaluation_loss_threshold",
        "score_field": score_field,
        "random_baselines": {"auc_roc": 0.5, "asr": 0.5,
                             "calibration_target_tpr_at_1fpr": 0.01,
                             "calibration_target_tpr_at_5fpr": 0.05,
                             "calibration_target_tpr_at_10fpr": 0.10},
        "attack_repeats": int(repeats),
        "attack_seed_start": int(attack_seed),
        "calibration_fraction_requested": float(calibration_fraction),
        "member_count": len(members),
        "nonmember_count": len(nonmembers),
        "member_source_groups": len({row["source_group_sha256"] for row in members}),
        "nonmember_source_groups": len({row["source_group_sha256"] for row in nonmembers}),
        "member_nonmember_source_group_intersection": 0,
        "calibration_evaluation_source_group_disjoint": True,
        "repeat_plan_sha256": json_sha256([
            row["split_plan_sha256"] for row in runs
        ]),
        "repeat_dispersion_note": (
            "Repeated attack-split SD is a robustness dispersion, not an independent-run "
            "confidence interval and not a replacement for training-seed uncertainty."
        ),
        "summary": summary,
        "runs": runs,
    }


def _macro_attack(client_results: Sequence[dict]) -> dict:
    output = {}
    for scope in ("local", "pooled"):
        output[scope] = {}
        for score in SCORE_FIELDS:
            output[scope][score] = {}
            for metric in (
                "auc_roc", "asr",
                "tpr_at_calibration_target_1fpr_threshold",
                "evaluation_fpr_at_calibration_target_1fpr_threshold",
                "tpr_at_calibration_target_5fpr_threshold",
                "evaluation_fpr_at_calibration_target_5fpr_threshold",
                "tpr_at_calibration_target_10fpr_threshold",
                "evaluation_fpr_at_calibration_target_10fpr_threshold",
            ):
                values = [
                    float(client["scopes"][scope]["attacks"][score]["summary"][metric]["mean"])
                    for client in client_results
                ]
                output[scope][score][metric] = {
                    "client_macro_mean": float(statistics.fmean(values)),
                    "client_sample_sd": (
                        float(statistics.stdev(values)) if len(values) > 1 else 0.0
                    ),
                }
    return output


def _write_summary_csv(report: dict, path: Path) -> None:
    fields = [
        "method", "client_id", "scope", "score", "member_count", "nonmember_count",
        "auc_mean", "auc_repeat_sd", "asr_mean", "asr_repeat_sd",
        "tpr_caltarget1_mean", "tpr_caltarget1_repeat_sd",
        "eval_fpr_caltarget1_mean", "eval_fpr_caltarget1_repeat_sd",
        "tpr_caltarget5_mean", "tpr_caltarget5_repeat_sd",
        "eval_fpr_caltarget5_mean", "eval_fpr_caltarget5_repeat_sd",
        "tpr_caltarget10_mean", "tpr_caltarget10_repeat_sd",
        "eval_fpr_caltarget10_mean", "eval_fpr_caltarget10_repeat_sd",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(temporary.parent, 0o700)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, method_result in report["methods"].items():
            for client in method_result["clients"]:
                for scope, scope_result in client["scopes"].items():
                    for score, attack in scope_result["attacks"].items():
                        summary = attack["summary"]
                        writer.writerow({
                            "method": method, "client_id": client["client_id"],
                            "scope": scope, "score": score,
                            "member_count": attack["member_count"],
                            "nonmember_count": attack["nonmember_count"],
                            "auc_mean": summary["auc_roc"]["mean"],
                            "auc_repeat_sd": summary["auc_roc"]["repeat_sample_sd"],
                            "asr_mean": summary["asr"]["mean"],
                            "asr_repeat_sd": summary["asr"]["repeat_sample_sd"],
                            "tpr_caltarget1_mean": summary["tpr_at_calibration_target_1fpr_threshold"]["mean"],
                            "tpr_caltarget1_repeat_sd": summary["tpr_at_calibration_target_1fpr_threshold"]["repeat_sample_sd"],
                            "eval_fpr_caltarget1_mean": summary["evaluation_fpr_at_calibration_target_1fpr_threshold"]["mean"],
                            "eval_fpr_caltarget1_repeat_sd": summary["evaluation_fpr_at_calibration_target_1fpr_threshold"]["repeat_sample_sd"],
                            "tpr_caltarget5_mean": summary["tpr_at_calibration_target_5fpr_threshold"]["mean"],
                            "tpr_caltarget5_repeat_sd": summary["tpr_at_calibration_target_5fpr_threshold"]["repeat_sample_sd"],
                            "eval_fpr_caltarget5_mean": summary["evaluation_fpr_at_calibration_target_5fpr_threshold"]["mean"],
                            "eval_fpr_caltarget5_repeat_sd": summary["evaluation_fpr_at_calibration_target_5fpr_threshold"]["repeat_sample_sd"],
                            "tpr_caltarget10_mean": summary["tpr_at_calibration_target_10fpr_threshold"]["mean"],
                            "tpr_caltarget10_repeat_sd": summary["tpr_at_calibration_target_10fpr_threshold"]["repeat_sample_sd"],
                            "eval_fpr_caltarget10_mean": summary["evaluation_fpr_at_calibration_target_10fpr_threshold"]["mean"],
                            "eval_fpr_caltarget10_repeat_sd": summary["evaluation_fpr_at_calibration_target_10fpr_threshold"]["repeat_sample_sd"],
                        })
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _write_summary_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Read-only MIA robustness audit\n",
        "Primary interpretation uses own-client (`local`) source-disjoint test images. "
        "The target is a label-aware white-box client-endpoint model, including its "
        "personalized factor when applicable; this is not an honest-but-curious server "
        "update-channel attack. Pooled results are a secondary resolution/sensitivity "
        "analysis. Lower attack AUC/balanced attack accuracy indicates less empirical "
        "membership leakage under this attack only, not a formal privacy guarantee. "
        "Repeated-split SD is not replicate uncertainty.\n",
        "| Method | Score | Local AUC | Local balanced accuracy (ASR) | Local TPR/FPR at cal-target 1% | Pooled AUC | Pooled balanced accuracy (ASR) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["methods"].items():
        macro = result["macro"]
        for score in SCORE_FIELDS:
            local = macro["local"][score]
            pooled = macro["pooled"][score]
            lines.append(
                f"| {method} | {score} | "
                f"{local['auc_roc']['client_macro_mean']:.4f} | "
                f"{local['asr']['client_macro_mean']:.4f} | "
                f"{local['tpr_at_calibration_target_1fpr_threshold']['client_macro_mean']:.4f}/"
                f"{local['evaluation_fpr_at_calibration_target_1fpr_threshold']['client_macro_mean']:.4f} | "
                f"{pooled['auc_roc']['client_macro_mean']:.4f} | "
                f"{pooled['asr']['client_macro_mean']:.4f} |"
            )
    lines.extend([
        "\n## Required interpretation checks\n",
        "- `trained_loss` is the conventional final-checkpoint loss-threshold MIA score. "
        "`ASR` is balanced evaluation accuracy after score direction and a Youden-J "
        "threshold are selected only on the held-out calibration subset.",
        "- `initial_loss` is a negative control. Above-random separation is evidence "
        "consistent with split-origin or sample-difficulty confounding, but its "
        "materiality requires replicate uncertainty.",
        "- `delta_loss` is an initialization-referenced loss-change diagnostic: trained "
        "loss minus exact fresh-initialization loss on the same image. It is not a "
        "standardized FL-MIA or a privacy guarantee.",
        "- Local nonmembers are the primary threat-model result. Pooled nonmembers improve "
        "low-FPR resolution but change the evaluation mixture.",
        "- This v1 audit does not implement covariate-matched attacks, confidence/entropy "
        "black-box attacks, or a source-group cluster bootstrap CI. Those are phase-2 analyses.",
    ])
    atomic_text_dump("\n".join(lines) + "\n", path)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS,
                        default=list(SUPPORTED_METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection_seed", type=int, default=420_042)
    parser.add_argument("--attack_seed", type=int, default=842_042)
    parser.add_argument("--attack_repeats", type=int, default=20)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--max_member_samples", type=int, default=1000)
    parser.add_argument("--max_local_nonmember_samples", type=int, default=1000)
    parser.add_argument("--max_pooled_nonmember_samples", type=int, default=2000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse only cache files whose complete cache key and record digest match")
    args = parser.parse_args(argv)
    if args.seed not in SUPPORTED_AUDIT_SEEDS:
        parser.error(
            "security_audit_v1 supports the frozen paired replicates seed 42, 43, or 44"
        )
    if len(args.methods) != len(set(args.methods)) or set(args.methods) != set(SUPPORTED_METHODS):
        parser.error(
            "security_audit_v1 requires exactly the four paired primary methods: "
            + " ".join(SUPPORTED_METHODS)
        )
    if args.attack_repeats != 20:
        parser.error("security_audit_v1 freezes --attack_repeats=20")
    if args.selection_seed != 420_042 or args.attack_seed != 842_042:
        parser.error(
            "security_audit_v1 freezes selection_seed=420042 and attack_seed=842042 "
            "for every training/partition seed"
        )
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration_fraction must be strictly between 0 and 1")
    if min(args.max_member_samples, args.max_local_nonmember_samples,
           args.max_pooled_nonmember_samples) < 4:
        parser.error("MIA sample limits must be at least 4")
    if not math.isclose(args.calibration_fraction, 0.5, rel_tol=0.0, abs_tol=0.0):
        parser.error("security_audit_v1 freezes --calibration_fraction=0.5")
    if (
        args.max_member_samples,
        args.max_local_nonmember_samples,
        args.max_pooled_nonmember_samples,
    ) != (1000, 1000, 2000):
        parser.error(
            "security_audit_v1 freezes member/local/pooled sample caps at "
            "1000/1000/2000"
        )
    return args


def _run_audit(argv=None) -> int:
    global _ACTIVE_PLAN_PATH, _ACTIVE_RUN_NONCE
    _ACTIVE_PLAN_PATH = None
    _ACTIVE_RUN_NONCE = None
    os.umask(0o077)
    args = _parse_args(argv)
    project_dir = Path(args.project_dir).resolve(strict=True)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    split_file = Path(args.split_file).resolve(strict=True)
    model_weights = Path(args.model_weights).resolve(strict=True)
    primary = resolve_primary_inputs(args.results_root, args.seed, args.methods)
    output_dir = validate_output_location(
        args.results_root, args.output_dir,
        [row["experiment_dir"] for row in primary.values()],
        seed=args.seed,
    )
    if output_dir.exists():
        symlinks = [str(path) for path in output_dir.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(
                "Audit output contains symbolic links and is unsafe to resume: "
                + ", ".join(symlinks[:10])
            )
    validate_resume_state(output_dir, resume=args.resume, dry_run=args.dry_run)

    protected_paths = [split_file, model_weights]
    for paths in primary.values():
        protected_paths.extend((
            paths["result"], paths["best_checkpoint"], paths["last_checkpoint"]
        ))
    before = snapshot_files(protected_paths)
    split_sha = before[str(split_file)]["sha256"]
    weights_sha = before[str(model_weights)]["sha256"]
    runtime_cache_root = output_dir / "runtime_cache"
    plan = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "dry_run" if args.dry_run else "execute",
        "read_only_primary": True,
        "seed": args.seed,
        "audit_instance": audit_output_name(args.seed),
        "methods": list(args.methods),
        "output_dir": str(output_dir),
        "runtime_cache_root": str(runtime_cache_root),
        "protected_inputs": before,
        "protocol": {
            "selection_seed": args.selection_seed,
            "attack_seed": args.attack_seed,
            "attack_repeats": args.attack_repeats,
            "calibration_fraction": args.calibration_fraction,
            "max_member_samples": args.max_member_samples,
            "max_local_nonmember_samples": args.max_local_nonmember_samples,
            "max_pooled_nonmember_samples": args.max_pooled_nonmember_samples,
            "calibration_evaluation_partition_unit": "source_group",
        },
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[PASS] Dry-run plan only; no output directory or primary artifact was written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    validate_existing_resume_plan(
        output_dir, plan, resume=args.resume, dry_run=args.dry_run
    )
    started = time.time()
    run_nonce = hashlib.sha256(
        f"{os.getpid()}\0{time.time_ns()}\0{split_sha}".encode("ascii")
    ).hexdigest()
    plan_path = output_dir / "audit_plan.json"
    atomic_json_dump(
        {**plan, "status": "running", "run_nonce": run_nonce}, plan_path
    )
    _ACTIVE_PLAN_PATH = plan_path
    _ACTIVE_RUN_NONCE = run_nonce
    # Keep library/runtime caches in the dedicated audit tree as well.  The
    # runner also sets PYTHONDONTWRITEBYTECODE; the in-process flag above covers
    # direct Python invocation.
    runtime_cache_root.mkdir(parents=True, exist_ok=True)
    os.chmod(runtime_cache_root, 0o700)
    # Override inherited cache locations: the audit may persist files only in
    # its dedicated tree, even when the interactive shell exports global paths.
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_cache_root / "ultralytics")
    os.environ["CUDA_CACHE_PATH"] = str(runtime_cache_root / "cuda")
    os.environ["TRITON_CACHE_DIR"] = str(runtime_cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime_cache_root / "torchinductor")
    split_info = load_json(split_file)
    data_info = _minimal_data_info(split_info, str(split_file))
    if data_info["split_manifest_sha256"] != split_sha:
        raise RuntimeError("Internal split SHA-256 mismatch")

    # Validate the immutable generated tree without invoking prepare_data, whose
    # image-verification cache may write beside the primary split manifest.
    from data.dataset import generated_yolo_tree_sha256, image_tree_sha256
    generated_before = generated_yolo_tree_sha256(data_info["yolo_dir"])
    expected_generated = split_info["metadata"].get("generated_yolo_tree_sha256")
    if generated_before != expected_generated:
        raise ValueError("Generated YOLO tree differs from the split manifest")
    # Full raw content verification is read-only and protects the source catalog.
    inventory = split_info["metadata"]["source_hash_inventory"]
    for split in ("train", "val", "test"):
        from data.dataset import canonicalize_aod4_categories, load_coco_annotations
        raw = load_coco_annotations(
            str(Path(args.data_root) / split / "_annotations.coco.json")
        )
        canonical, _ = canonicalize_aod4_categories(raw)
        actual_tree = image_tree_sha256(canonical, str(Path(args.data_root) / split))
        if actual_tree != inventory["per_split_image_tree_sha256"][split]:
            raise ValueError(f"{split}: raw image tree differs from immutable inventory")

    catalogs, selected = _load_catalog(split_info, args.data_root)
    candidate_sets = _candidate_sets(
        split_info, catalogs, selected, selection_seed=args.selection_seed,
        max_member=args.max_member_samples,
        max_local_nonmember=args.max_local_nonmember_samples,
        max_pooled_nonmember=args.max_pooled_nonmember_samples,
    )

    # Check all result metadata before importing/loading any checkpoint tensors.
    primary_payloads = {}
    for method, paths in primary.items():
        payload = load_json(paths["result"])
        _strict_primary_result(
            payload, method=method, seed=args.seed, split_sha256=split_sha,
            best_checkpoint=paths["best_checkpoint"],
        )
        architecture = payload.get("architecture", {})
        if architecture.get("model_weight_sha256") != weights_sha:
            raise ValueError(f"{method}: primary model-weight SHA-256 mismatch")
        primary_payloads[method] = payload

    from models.rtdetr_lora import RTDETRLoRA, _tensor_state_sha256
    from trainers.fl_server import _apply_checkpoint, _load_checkpoint, _new_client_models

    checkpoint_payloads = {
        method: _load_checkpoint(paths["best_checkpoint"])
        for method, paths in primary.items()
    }
    method_args = {}
    primary_training_protocol_sha256 = {}
    initial_digests = set()
    for method, checkpoint in checkpoint_payloads.items():
        compatibility = checkpoint.get("compatibility")
        if not isinstance(compatibility, dict):
            raise ValueError(f"{method}: checkpoint lacks strict compatibility metadata")
        primary_training_protocol_sha256[method] = (
            _validate_primary_checkpoint_compatibility(
                compatibility, method=method, seed=args.seed
            )
        )
        resolved_args = _args_from_checkpoint_compatibility(
            compatibility, data_root=args.data_root, split_file=str(split_file),
            model_weights=str(model_weights), device=args.device,
        )
        method_args[method] = resolved_args
        result_selection = primary_payloads[method]["selection"]
        if (
            checkpoint.get("selection") != result_selection["criterion"]
            or int(checkpoint.get("round", -1)) != int(result_selection["round"])
            or checkpoint.get("split_manifest_sha256") != split_sha
        ):
            raise ValueError(f"{method}: best checkpoint/result selection mismatch")
        initial_digest = _validate_architecture_provenance(
            method=method,
            checkpoint_architecture=checkpoint.get("architecture"),
            primary_architecture=primary_payloads[method].get("architecture"),
            model_weight_sha256=weights_sha,
        )
        initial_digests.add(initial_digest)
    if len(initial_digests) != 1:
        raise ValueError(
            "Primary methods did not start from the same target initialization: "
            f"{sorted(initial_digests)}"
        )
    expected_initial_digest = next(iter(initial_digests))

    cache_root = output_dir / "loss_records"
    initial_cache_root = cache_root / "initial_control"
    method_results = {}
    control_state_digests = {}
    pairing_gate = {}
    repeat_plan_gate = defaultdict(set)
    initial_model = None
    try:
        for method in args.methods:
            print(f"\n[AUDIT] Restoring {method} from best_federated.pt")
            resolved_args = method_args[method]
            # Build a method-specific fresh control.  The common pre-injection
            # target digest proves the same RT-DETR target initialization, while
            # this post-injection state digest also binds the exact LoRA wrapper
            # execution path used by the corresponding trained method.
            _set_seed(args.seed)
            initial_model = RTDETRLoRA(
                resolved_args, class_names=data_info["class_names"]
            )
            fresh_initial_digest = initial_model.initial_target_state_sha256
            if fresh_initial_digest != expected_initial_digest:
                raise ValueError(
                    f"{method}: fresh target initialization digest differs from training; "
                    "method-specific split-origin control and delta loss fail closed: "
                    f"fresh={fresh_initial_digest}, expected={expected_initial_digest}"
                )
            initial_model_state_before = _tensor_state_sha256(
                initial_model.model.state_dict()
            )
            control_state_digests[method] = initial_model_state_before
            _set_seed(args.seed)
            client_models = _new_client_models(resolved_args, data_info)
            # A fresh pre-application digest gate is required for every architecture.
            if client_models[0].initial_target_state_sha256 != expected_initial_digest:
                raise ValueError(f"{method}: fresh initial target digest drift")
            _apply_checkpoint(
                checkpoint_payloads[method], client_models, resolved_args, data_info
            )
            expected_role = {
                "full_ft": None, "lora": None,
                "fedsa_lora": "B", "fixed_share_b_lora": "A",
            }[method]
            observed_roles = [model.local_personalized_factor_role() for model in client_models]
            if observed_roles != [expected_role] * 3:
                raise RuntimeError(
                    f"{method}: personalized role mismatch {observed_roles}, "
                    f"expected {expected_role!r}"
                )
            reconstructed_digests = [
                _tensor_state_sha256(model.model.state_dict()) for model in client_models
            ]
            clients_result = []
            for client_id, model in enumerate(client_models):
                member_candidates = candidate_sets[client_id]["member"]
                scope_candidates = {
                    "local": candidate_sets[client_id]["local_nonmember"],
                    "pooled": candidate_sets[client_id]["pooled_nonmember"],
                }
                scope_results = {}
                for scope, nonmember_candidates in scope_candidates.items():
                    data_yaml_nonmember = (
                        data_info["client_yamls"][client_id]
                        if scope == "local" else data_info["full_yaml"]
                    )
                    selections = {
                        "member": (member_candidates, data_info["client_yamls"][client_id], "train"),
                        "nonmember": (nonmember_candidates, data_yaml_nonmember, "test"),
                    }
                    merged_by_membership = {}
                    for membership, (candidates, data_yaml, split) in selections.items():
                        sample_digest = json_sha256(
                            [row["sample_id_sha256"] for row in candidates]
                        )
                        pairing_gate.setdefault((client_id, scope, membership), set()).add(
                            sample_digest
                        )
                        cache_scope = "member_shared" if membership == "member" else scope
                        initial_client_id = (
                            None if membership == "nonmember" and scope == "pooled"
                            else client_id
                        )
                        initial_key = {
                            "kind": "method_specific_fresh_target_initialization_control",
                            "control_method": method,
                            "initial_target_tensor_state_sha256": fresh_initial_digest,
                            "control_model_state_sha256": initial_model_state_before,
                            "model_weight_sha256": weights_sha,
                            "split_manifest_sha256": split_sha,
                            "client_id": initial_client_id,
                            "scope": cache_scope,
                            "membership": membership,
                            "sample_ids_sha256": sample_digest,
                            "sample_count": len(candidates),
                            "img_size": int(resolved_args.img_size),
                            "batch_size": 1,
                        }
                        initial_name = (
                            "pooled_global_nonmember.json"
                            if membership == "nonmember" and scope == "pooled"
                            else f"client_{client_id}_{cache_scope}_{membership}.json"
                        )
                        initial_path = initial_cache_root / method / initial_name
                        initial_records = _cache_load_or_extract(
                            initial_path, initial_key, resume=args.resume,
                            extractor=lambda c=candidates, y=data_yaml, s=split: _extract_losses(
                                initial_model, y, s, resolved_args, c
                            ),
                        )
                        trained_key = {
                            "kind": "trained_validation_selected_personalized_model",
                            "method": method,
                            "checkpoint_sha256": before[primary[method]["best_checkpoint"]]["sha256"],
                            "reconstructed_client_state_sha256": reconstructed_digests[client_id],
                            "split_manifest_sha256": split_sha,
                            "client_id": client_id,
                            "scope": cache_scope,
                            "membership": membership,
                            "sample_ids_sha256": sample_digest,
                            "sample_count": len(candidates),
                            "img_size": int(resolved_args.img_size),
                            "batch_size": 1,
                        }
                        trained_path = cache_root / method / (
                            f"client_{client_id}_{cache_scope}_{membership}.json"
                        )
                        trained_records = _cache_load_or_extract(
                            trained_path, trained_key, resume=args.resume,
                            extractor=lambda c=candidates, y=data_yaml, s=split, m=model: _extract_losses(
                                m, y, s, resolved_args, c
                            ),
                        )
                        merged_by_membership[membership] = _merge_losses(
                            trained_records, initial_records
                        )
                    member_records = merged_by_membership["member"]
                    nonmember_records = merged_by_membership["nonmember"]
                    attacks = {
                        score: repeated_attack(
                            member_records, nonmember_records, score_field=score,
                            repeats=args.attack_repeats,
                            attack_seed=args.attack_seed + 10_007 * client_id,
                            calibration_fraction=args.calibration_fraction,
                        )
                        for score in SCORE_FIELDS
                    }
                    for score, attack in attacks.items():
                        repeat_plan_gate[(client_id, scope)].add(
                            attack["repeat_plan_sha256"]
                        )
                    scope_results[scope] = {
                        "member_sample_ids_sha256": json_sha256(
                            [row["sample_id_sha256"] for row in member_records]
                        ),
                        "nonmember_sample_ids_sha256": json_sha256(
                            [row["sample_id_sha256"] for row in nonmember_records]
                        ),
                        "source_policy": (
                            "own_client_test_excluding_any_global_train_source_group"
                            if scope == "local"
                            else "pooled_official_test_excluding_any_global_train_source_group"
                        ),
                        "attacks": attacks,
                    }
                clients_result.append({
                    "client_id": client_id,
                    "reconstructed_model_state_sha256": reconstructed_digests[client_id],
                    "local_personalized_factor_role": observed_roles[client_id],
                    "scopes": scope_results,
                })
            reconstructed_after = [
                _tensor_state_sha256(model.model.state_dict()) for model in client_models
            ]
            if reconstructed_after != reconstructed_digests:
                raise RuntimeError(
                    f"{method}: model state changed during no-grad loss extraction; "
                    f"before={reconstructed_digests}, after={reconstructed_after}"
                )
            initial_model_state_after = _tensor_state_sha256(
                initial_model.model.state_dict()
            )
            if initial_model_state_after != initial_model_state_before:
                raise RuntimeError(
                    f"{method}: fresh control state changed during loss extraction; "
                    f"before={initial_model_state_before}, after={initial_model_state_after}"
                )
            method_results[method] = {
                "primary_result_sha256": before[primary[method]["result"]]["sha256"],
                "best_checkpoint_sha256": before[primary[method]["best_checkpoint"]]["sha256"],
                "primary_training_protocol_sha256": (
                    primary_training_protocol_sha256[method]
                ),
                "architecture_manifest_sha256": json_sha256(
                    checkpoint_payloads[method]["architecture"]
                ),
                "selection_round": int(checkpoint_payloads[method]["round"]),
                "federated_payload_policy": checkpoint_payloads[method].get(
                    "federated_payload_policy"
                ),
                "fresh_control_model_state_sha256": initial_model_state_before,
                "clients": clients_result,
                "macro": _macro_attack(clients_result),
                "post_extraction_model_state_gate": "pass",
            }
            del client_models
            del initial_model
            initial_model = None
            import torch
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        if initial_model is not None:
            del initial_model

    pairing_failures = {
        f"client_{client_id}/{scope}/{membership}": sorted(digests)
        for (client_id, scope, membership), digests in pairing_gate.items()
        if len(digests) != 1
    }
    if pairing_failures:
        raise RuntimeError(f"Methods used different attack samples: {pairing_failures}")
    repeat_plan_failures = {
        f"client_{client_id}/{scope}": sorted(digests)
        for (client_id, scope), digests in repeat_plan_gate.items()
        if len(digests) != 1
    }
    if repeat_plan_failures:
        raise RuntimeError(
            "Methods/scores used different repeated attack assignments: "
            f"{repeat_plan_failures}"
        )

    generated_after = generated_yolo_tree_sha256(data_info["yolo_dir"])
    if generated_after != generated_before:
        raise RuntimeError("Generated YOLO tree changed during the read-only audit")
    after = snapshot_files(protected_paths)
    changes = changed_snapshot(before, after)
    integrity = {
        "schema_version": 1,
        "status": "pass" if not changes else "fail",
        "read_only_primary_gate": not changes,
        "before": before,
        "after": after,
        "changes": changes,
        "generated_yolo_tree_sha256_before": generated_before,
        "generated_yolo_tree_sha256_after": generated_after,
    }
    atomic_json_dump(integrity, output_dir / "integrity_manifest.json")
    if changes:
        raise RuntimeError(f"Protected primary artifacts changed: {changes}")

    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "audit_name": "security_audit_v1",
        "audit_instance": audit_output_name(args.seed),
        "read_only_primary_gate": "pass",
        "seed": args.seed,
        "split_manifest_sha256": split_sha,
        "model_weight_sha256": weights_sha,
        "fresh_initial_target_tensor_state_sha256": expected_initial_digest,
        "method_specific_fresh_control_state_sha256": control_state_digests,
        "exact_initial_digest_gate": "pass",
        "checkpoint_primary_architecture_manifest_gate": "pass",
        "frozen_primary_training_protocol_gate": "pass",
        "post_extraction_initial_model_state_gate": "pass",
        "sample_pairing_across_methods_gate": "pass",
        "repeated_attack_plan_pairing_gate": "pass",
        "protocol": plan["protocol"],
        "threat_model": (
            "label-aware white-box client-endpoint audit of the final personalized model; "
            "image-level ground-truth RT-DETR loss; source-group metadata available to "
            "the evaluator; not an honest-but-curious server update-channel attack"
        ),
        "interpretation": (
            "Empirical membership-leakage robustness audit, not a confidentiality or "
            "differential-privacy guarantee."
        ),
        "metric_definitions": {
            "auc_roc": (
                "Threshold-independent ROC area on the balanced held-out evaluation "
                "member/nonmember set."
            ),
            "asr": (
                "Balanced attack accuracy on the held-out evaluation set after score "
                "direction and a Youden-J threshold are selected using only the "
                "calibration subset."
            ),
            "low_fpr": (
                "Evaluation TPR and achieved evaluation FPR after fixing a threshold "
                "whose calibration FPR is at most the requested target."
            ),
        },
        "primary_scope": "local",
        "secondary_scope": "pooled",
        "controls": {
            "initial_loss": (
                "Fresh exact-digest-matched four-class target initialization used as a "
                "negative control. Above-random separation is evidence consistent with, "
                "but does not by itself establish, split-origin/sample-difficulty confounding."
            ),
            "delta_loss": (
                "Initialization-referenced loss-change diagnostic: trained_loss minus "
                "paired initial_loss for the same image; not a standardized FL-MIA score."
            ),
        },
        "limitations": [
            "Repeated attack-split SD is not training-seed uncertainty.",
            "No covariate-matched, confidence/entropy black-box, update-level, or DP attack is included.",
            "No cluster-bootstrap confidence interval is included in v1; calibration/evaluation splits are source-group atomic.",
            "A single-seed audit must not be generalized; aggregate seeds 42--44 as paired training/partition-seed replicates.",
        ],
        "methods": method_results,
        "elapsed_seconds": float(time.time() - started),
    }
    # Deliberately avoid the *_results.json suffix: aggregate_results.py uses
    # that suffix to discover primary training outputs recursively.
    atomic_json_dump(report, output_dir / "audit_report.json")
    _write_summary_csv(report, output_dir / "audit_summary.csv")
    _write_summary_markdown(report, output_dir / "summary.md")
    output_hashes = snapshot_files([
        output_dir / "audit_plan.json", output_dir / "audit_report.json",
        output_dir / "audit_summary.csv", output_dir / "summary.md",
    ])
    integrity["audit_outputs"] = output_hashes
    atomic_json_dump(integrity, output_dir / "integrity_manifest.json")
    print(f"[PASS] Read-only primary integrity gate: {len(before)} protected files unchanged")
    print("[PASS] Exact fresh-initialization and cross-method sample-pairing gates")
    print(f"[Results] {output_dir / 'audit_report.json'}")
    print(f"[Summary] {output_dir / 'summary.md'}")
    return 0


def _write_failure_integrity_manifest(argv, error: BaseException) -> None:
    """Best-effort final SHA gate for every failure after audit_plan creation."""
    try:
        del argv
        if _ACTIVE_PLAN_PATH is None or _ACTIVE_RUN_NONCE is None:
            return
        plan_path = _ACTIVE_PLAN_PATH
        if not plan_path.is_file():
            return
        plan = load_json(plan_path)
        if plan.get("run_nonce") != _ACTIVE_RUN_NONCE:
            return
        output_dir = plan_path.parent
        before = plan.get("protected_inputs")
        if not isinstance(before, dict) or not before:
            return
        after = snapshot_files_best_effort(before)
        changes = changed_snapshot(before, after)
        atomic_json_dump({
            "schema_version": 1,
            "status": "fail",
            "read_only_primary_gate": not changes,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "before": before,
            "after": after,
            "changes": changes,
        }, output_dir / "integrity_manifest.json")
    except Exception as integrity_error:  # pragma: no cover - best-effort crash path
        print(
            "[CRITICAL] Could not complete failure-path primary integrity gate: "
            f"{type(integrity_error).__name__}: {integrity_error}",
            file=sys.stderr,
        )


def main(argv=None) -> int:
    try:
        return _run_audit(argv)
    except BaseException as error:
        _write_failure_integrity_manifest(argv, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
