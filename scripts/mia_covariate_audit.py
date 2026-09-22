#!/usr/bin/env python3
"""Read-only covariate-matched reanalysis of frozen v1 MIA loss caches.

This phase-2 audit never loads a model. For each repeated attack it first makes
source-group-atomic calibration/evaluation partitions, then performs
outcome-blind coarsened exact matching independently inside both partitions.
The exact same sample plans are reused by every method and loss score.

Only the existing seed-42/43/44 v1 local member/nonmember caches are read.
Primary artifacts and v1 artifacts are SHA-256 checked before and after, and a
completed output is atomically published to a new immutable directory.

This is an exploratory post-hoc sensitivity analysis motivated by the v1
initial-loss negative control; it is not a preregistered confirmatory attack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SEEDS = (42, 43, 44)
METHODS = ("full_ft", "lora", "fedsa_lora", "fixed_share_b_lora")
CLASS_NAMES = ("airplane", "bird", "drone", "helicopter")
SCORES = ("trained_loss", "initial_loss", "delta_loss")
# A matched local half-partition can contain fewer than 100 nonmembers. A 1%
# operating point would then be a zero-false-positive rule rather than a
# resolved 1% estimate, so phase 2 reports only 5% and 10% thresholds.
FPR_TARGETS = (0.05, 0.10)
OUTPUT_NAME = "security_audit_v2_covariate_multiseed"
MATCH_SEED = 942_042
ATTACK_SEED = 1_842_042
ATTACK_REPEATS = 20
CALIBRATION_FRACTION = 0.5
MIN_MATCHED_PAIRS_PER_PARTITION = 64
MIN_RETENTION_PER_PARTITION = 0.25
MAX_ABS_SMD = 0.10
SCHEMA_VERSION = 2

RECORD_METADATA_FIELDS = (
    "sample_id_sha256",
    "image_content_sha256",
    "source_group_sha256",
    "split",
    "width",
    "height",
    "object_count",
    "background",
    "class_counts",
    "classes",
    "bbox_area_ratio_mean",
    "bbox_area_ratio_min",
    "bbox_area_ratio_max",
)
ATTACK_METRICS = (
    "auc_roc",
    "asr",
    "tpr_at_calibration_target_5fpr_threshold",
    "evaluation_fpr_at_calibration_target_5fpr_threshold",
    "tpr_at_calibration_target_10fpr_threshold",
    "evaluation_fpr_at_calibration_target_10fpr_threshold",
)


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


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _atomic_json(payload, path: Path) -> None:
    _atomic_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        path,
    )


def _atomic_csv(rows: Sequence[Mapping], fields: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _load_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Audit input must be a regular non-symlink file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot(paths: Iterable[os.PathLike | str]) -> dict:
    output = {}
    for raw in sorted({str(Path(path).absolute()) for path in paths}):
        original = Path(raw)
        if original.is_symlink() or not original.is_file():
            raise ValueError(f"Protected input is missing or a symbolic link: {original}")
        resolved = original.resolve(strict=True)
        key = str(resolved)
        if key in output:
            continue
        output[key] = {
            "path": key,
            "size_bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }
    return output


def _changed(before: Mapping, after: Mapping) -> list[dict]:
    return [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]


def _audit_dir_name(seed: int) -> str:
    return "security_audit_v1" if int(seed) == 42 else f"security_audit_v1_seed{seed}"


def validate_output_location(results_root: Path, output_dir: Path) -> Path:
    root = results_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    expected = root / OUTPUT_NAME
    if output != expected:
        raise ValueError(f"Phase-2 audit writes only to {expected}; received {output}")
    if output.exists():
        raise FileExistsError(
            f"Completed/ambiguous phase-2 output is immutable: {output}. "
            "Preserve it and use a new versioned protocol instead of overwriting."
        )
    return output


def _cache_path(
    audit_dir: Path, method: str, client_id: int, membership: str
) -> tuple[Path, Path]:
    if membership == "member":
        name = f"client_{client_id}_member_shared_member.json"
    elif membership == "nonmember":
        name = f"client_{client_id}_local_nonmember.json"
    else:  # pragma: no cover - internal caller contract
        raise ValueError(membership)
    root = audit_dir / "loss_records"
    return root / method / name, root / "initial_control" / method / name


def _client_report(report: Mapping, method: str, client_id: int) -> Mapping:
    rows = [
        row
        for row in report["methods"][method]["clients"]
        if int(row.get("client_id", -1)) == int(client_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"Missing unique v1 client report: {method}/client-{client_id}")
    return rows[0]


def _expected_cache_key(
    report: Mapping,
    *,
    method: str,
    client_id: int,
    membership: str,
    initial: bool,
) -> dict:
    client = _client_report(report, method, client_id)
    local = client["scopes"]["local"]
    attack = local["attacks"]["trained_loss"]
    count_key = "member_count" if membership == "member" else "nonmember_count"
    digest_key = (
        "member_sample_ids_sha256"
        if membership == "member"
        else "nonmember_sample_ids_sha256"
    )
    scope = "member_shared" if membership == "member" else "local"
    common = {
        "split_manifest_sha256": report["split_manifest_sha256"],
        "client_id": int(client_id),
        "scope": scope,
        "membership": membership,
        "sample_ids_sha256": local[digest_key],
        "sample_count": int(attack[count_key]),
        "img_size": 640,
        "batch_size": 1,
    }
    if initial:
        control_digest = report["methods"][method]["fresh_control_model_state_sha256"]
        top_control = report["method_specific_fresh_control_state_sha256"][method]
        if control_digest != top_control:
            raise ValueError(f"v1 fresh-control state digest mismatch for {method}")
        return {
            "kind": "method_specific_fresh_target_initialization_control",
            "control_method": method,
            "initial_target_tensor_state_sha256": report[
                "fresh_initial_target_tensor_state_sha256"
            ],
            "control_model_state_sha256": control_digest,
            "model_weight_sha256": report["model_weight_sha256"],
            **common,
        }
    return {
        "kind": "trained_validation_selected_personalized_model",
        "method": method,
        "checkpoint_sha256": report["methods"][method]["best_checkpoint_sha256"],
        "reconstructed_client_state_sha256": client[
            "reconstructed_model_state_sha256"
        ],
        **common,
    }


def _validate_record(row: Mapping, *, membership: str, path: Path) -> None:
    required = set(RECORD_METADATA_FIELDS) | {"loss", "loss_components"}
    forbidden = {"file_name", "image_path", "image_id", "_file_name", "_image_id"}
    if not isinstance(row, dict) or not required.issubset(row) or forbidden & set(row):
        raise ValueError(f"Incomplete or path-bearing loss record in {path}")
    for key in ("sample_id_sha256", "image_content_sha256", "source_group_sha256"):
        if not _is_sha256(row[key]):
            raise ValueError(f"Invalid {key} in {path}")
    expected_split = "train" if membership == "member" else "test"
    if row["split"] != expected_split:
        raise ValueError(
            f"{path}: {membership} record must have split={expected_split}, "
            f"received {row['split']!r}"
        )
    if int(row["width"]) <= 0 or int(row["height"]) <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    counts = row["class_counts"]
    if not isinstance(counts, dict) or any(
        name not in CLASS_NAMES
        or isinstance(value, bool)
        or int(value) != value
        or int(value) <= 0
        for name, value in counts.items()
    ):
        raise ValueError(f"Invalid class_counts in {path}")
    normalized_counts = {str(name): int(value) for name, value in counts.items()}
    object_count = int(row["object_count"])
    if object_count < 0 or object_count != sum(normalized_counts.values()):
        raise ValueError(f"object_count/class_counts mismatch in {path}")
    if bool(row["background"]) != (object_count == 0):
        raise ValueError(f"background/object_count mismatch in {path}")
    if row["classes"] != sorted(normalized_counts):
        raise ValueError(f"classes/class_counts mismatch in {path}")
    minimum = float(row["bbox_area_ratio_min"])
    mean = float(row["bbox_area_ratio_mean"])
    maximum = float(row["bbox_area_ratio_max"])
    # The frozen v1 producer divided the raw COCO bbox area by image area.
    # It did not clip the stored bbox before this calculation, so a valid
    # producer-compatible ratio can exceed 1 for an out-of-frame raw box.
    if not all(math.isfinite(value) and value >= 0.0 for value in (minimum, mean, maximum)):
        raise ValueError(f"Invalid bbox-to-image area ratios in {path}")
    if object_count == 0:
        if (minimum, mean, maximum) != (0.0, 0.0, 0.0):
            raise ValueError(f"Background record has nonzero box area in {path}")
    elif not minimum <= mean <= maximum:
        raise ValueError(f"Box-area min/mean/max ordering is invalid in {path}")
    if not math.isfinite(float(row["loss"])):
        raise ValueError(f"Non-finite loss in {path}")
    if not isinstance(row["loss_components"], dict):
        raise ValueError(f"Invalid loss_components in {path}")


def _load_cache(path: Path, expected_key: Mapping) -> list[dict]:
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("cache_key") != dict(expected_key)
        or not isinstance(records, list)
        or payload.get("records_sha256") != json_sha256(records)
    ):
        raise ValueError(f"Invalid, stale, or provenance-mismatched v1 cache: {path}")
    ids = [row.get("sample_id_sha256") for row in records if isinstance(row, dict)]
    if (
        len(ids) != len(records) == int(expected_key["sample_count"])
        or len(ids) != len(set(ids))
        or json_sha256(ids) != expected_key["sample_ids_sha256"]
    ):
        raise ValueError(f"Invalid sample identity envelope in v1 cache: {path}")
    membership = str(expected_key["membership"])
    for row in records:
        _validate_record(row, membership=membership, path=path)
    return records


def _merge_records(trained: Sequence[dict], initial: Sequence[dict]) -> list[dict]:
    initial_by_id = {row["sample_id_sha256"]: row for row in initial}
    if len(initial_by_id) != len(initial):
        raise ValueError("Initial cache has duplicate sample IDs")
    if {row["sample_id_sha256"] for row in trained} != set(initial_by_id):
        raise ValueError("Trained and initial caches do not contain identical samples")
    output = []
    for trained_row in trained:
        sample_id = trained_row["sample_id_sha256"]
        initial_row = initial_by_id[sample_id]
        for field in RECORD_METADATA_FIELDS:
            if trained_row[field] != initial_row[field]:
                raise ValueError(f"Trained/initial metadata mismatch for {sample_id}: {field}")
        trained_loss = float(trained_row["loss"])
        initial_loss = float(initial_row["loss"])
        delta_loss = trained_loss - initial_loss
        if not math.isfinite(delta_loss):
            raise ValueError(f"Non-finite delta loss for {sample_id}")
        output.append({
            **{field: trained_row[field] for field in RECORD_METADATA_FIELDS},
            "trained_loss": trained_loss,
            "initial_loss": initial_loss,
            "delta_loss": delta_loss,
        })
    return output


def metadata_sha256(rows: Sequence[Mapping]) -> str:
    projected = [
        {field: row[field] for field in RECORD_METADATA_FIELDS}
        for row in sorted(rows, key=lambda item: str(item["sample_id_sha256"]))
    ]
    return json_sha256(projected)


def _stable_key(seed: int, label: str, identifier: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{label}\0{identifier}".encode("ascii")
    ).hexdigest()


def _area_bin(value: float) -> int:
    value = float(value)
    if value <= 0.0:
        return -99
    return max(-24, min(0, int(math.floor(math.log2(value)))))


def covariate_stratum(row: Mapping) -> tuple:
    counts = row["class_counts"]
    return (
        bool(row["background"]),
        tuple(min(3, int(counts.get(name, 0))) for name in CLASS_NAMES),
        min(6, int(row["object_count"])),
        _area_bin(float(row["bbox_area_ratio_mean"])),
    )


def _feature_values(rows: Sequence[Mapping]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {
        "background": [],
        "log1p_object_count": [],
        "log_bbox_area_mean": [],
        "log_image_pixel_area": [],
        "log_aspect_ratio": [],
    }
    for name in CLASS_NAMES:
        features[f"presence_{name}"] = []
        features[f"count_{name}"] = []
    for row in rows:
        counts = row["class_counts"]
        background = bool(row["background"])
        width, height = float(row["width"]), float(row["height"])
        features["background"].append(float(background))
        features["log1p_object_count"].append(math.log1p(float(row["object_count"])))
        area = float(row["bbox_area_ratio_mean"])
        features["log_bbox_area_mean"].append(
            0.0 if background else math.log(max(area, 1e-12))
        )
        features["log_image_pixel_area"].append(math.log(width * height))
        features["log_aspect_ratio"].append(math.log(width / height))
        for name in CLASS_NAMES:
            count = float(counts.get(name, 0))
            features[f"presence_{name}"].append(float(count > 0))
            features[f"count_{name}"].append(count)
    return features


def _smd(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        raise ValueError("SMD requires two nonempty samples")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    left_var = statistics.variance(left) if len(left) > 1 else 0.0
    right_var = statistics.variance(right) if len(right) > 1 else 0.0
    pooled = math.sqrt((left_var + right_var) / 2.0)
    if pooled <= 1e-15:
        return 0.0 if math.isclose(left_mean, right_mean, abs_tol=1e-15) else math.copysign(
            2.0, left_mean - right_mean
        )
    return float((left_mean - right_mean) / pooled)


def balance_diagnostics(members: Sequence[Mapping], nonmembers: Sequence[Mapping]) -> dict:
    member_features = _feature_values(members)
    nonmember_features = _feature_values(nonmembers)
    smd = {
        name: _smd(member_features[name], nonmember_features[name])
        for name in sorted(member_features)
    }
    return {
        "standardized_mean_differences": smd,
        "max_abs_smd": max((abs(value) for value in smd.values()), default=0.0),
    }


def coarsened_exact_match(
    members: Sequence[dict],
    nonmembers: Sequence[dict],
    *,
    seed: int = MATCH_SEED,
    minimum_pairs: int = MIN_MATCHED_PAIRS_PER_PARTITION,
    minimum_retention: float = MIN_RETENTION_PER_PARTITION,
    maximum_abs_smd: float = MAX_ABS_SMD,
) -> dict:
    member_strata: dict[tuple, list[dict]] = defaultdict(list)
    nonmember_strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in members:
        member_strata[covariate_stratum(row)].append(row)
    for row in nonmembers:
        nonmember_strata[covariate_stratum(row)].append(row)
    pairs = []
    shared = sorted(set(member_strata) & set(nonmember_strata), key=repr)
    for stratum in shared:
        left = sorted(
            member_strata[stratum],
            key=lambda row: _stable_key(seed, "member", row["sample_id_sha256"]),
        )
        right = sorted(
            nonmember_strata[stratum],
            key=lambda row: _stable_key(seed, "nonmember", row["sample_id_sha256"]),
        )
        pairs.extend(
            (member["sample_id_sha256"], nonmember["sample_id_sha256"])
            for member, nonmember in zip(left, right)
        )
    pairs.sort(key=lambda pair: _stable_key(seed, "pair", "\0".join(pair)))
    member_map = {row["sample_id_sha256"]: row for row in members}
    nonmember_map = {row["sample_id_sha256"]: row for row in nonmembers}
    matched_members = [member_map[left] for left, _ in pairs]
    matched_nonmembers = [nonmember_map[right] for _, right in pairs]
    pre = balance_diagnostics(members, nonmembers)
    post = balance_diagnostics(matched_members, matched_nonmembers) if pairs else {
        "standardized_mean_differences": {},
        "max_abs_smd": 2.0,
    }
    smaller = min(len(members), len(nonmembers))
    retention = len(pairs) / smaller if smaller else 0.0
    if len(pairs) < int(minimum_pairs):
        raise ValueError(
            f"Covariate matching retained {len(pairs)} pairs; minimum is {minimum_pairs}"
        )
    if retention < float(minimum_retention):
        raise ValueError(
            f"Covariate matching retention {retention:.4f} is below "
            f"{minimum_retention:.2f}"
        )
    if float(post["max_abs_smd"]) > float(maximum_abs_smd):
        raise ValueError(
            f"Post-match max |SMD|={post['max_abs_smd']:.4f} exceeds "
            f"the frozen {maximum_abs_smd:.2f} gate"
        )
    return {
        "pairs": pairs,
        "member_count_before": len(members),
        "nonmember_count_before": len(nonmembers),
        "matched_pairs": len(pairs),
        "retention_of_smaller_pool": retention,
        "shared_strata": len(shared),
        "pre_match_balance": pre,
        "post_match_balance": post,
        "match_plan_sha256": json_sha256(pairs),
    }


def _validate_pool_pair(members: Sequence[dict], nonmembers: Sequence[dict]) -> None:
    if min(len(members), len(nonmembers)) < 2 * MIN_MATCHED_PAIRS_PER_PARTITION:
        raise ValueError("Raw member/nonmember pools are too small for two matched partitions")
    for label, field in (
        ("sample", "sample_id_sha256"),
        ("source", "source_group_sha256"),
        ("content", "image_content_sha256"),
    ):
        member_values = {row[field] for row in members}
        nonmember_values = {row[field] for row in nonmembers}
        if len(member_values) != len(members) and field == "sample_id_sha256":
            raise ValueError(f"Duplicate member {label} identifiers")
        if len(nonmember_values) != len(nonmembers) and field == "sample_id_sha256":
            raise ValueError(f"Duplicate nonmember {label} identifiers")
        overlap = member_values & nonmember_values
        if overlap:
            raise ValueError(f"Member/nonmember {label} overlap: {len(overlap)}")
    if min(
        len({row["source_group_sha256"] for row in members}),
        len({row["source_group_sha256"] for row in nonmembers}),
    ) < 2:
        raise ValueError("Each membership pool needs at least two source groups")


def source_group_atomic_split(
    rows: Sequence[dict], *, fraction: float, seed: int, label: str
) -> tuple[list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group_sha256"])].append(row)
    ordered = sorted(groups.items(), key=lambda item: _stable_key(seed, label, item[0]))
    if len(ordered) < 2:
        raise ValueError("Source-group-atomic split needs at least two source groups")
    target = len(rows) * float(fraction)
    cumulative = 0
    candidates = []
    for cutoff in range(1, len(ordered)):
        cumulative += len(ordered[cutoff - 1][1])
        candidates.append((abs(cumulative - target), cutoff))
    cutoff = min(candidates)[1]
    calibration = [row for _, group in ordered[:cutoff] for row in group]
    evaluation = [row for _, group in ordered[cutoff:] for row in group]
    cal_groups = {row["source_group_sha256"] for row in calibration}
    eval_groups = {row["source_group_sha256"] for row in evaluation}
    if not calibration or not evaluation or cal_groups & eval_groups:
        raise RuntimeError("Invalid source-group-atomic calibration/evaluation split")
    public = {
        "calibration_count": len(calibration),
        "evaluation_count": len(evaluation),
        "calibration_source_groups": len(cal_groups),
        "evaluation_source_groups": len(eval_groups),
        "calibration_ids_sha256": json_sha256(
            sorted(row["sample_id_sha256"] for row in calibration)
        ),
        "evaluation_ids_sha256": json_sha256(
            sorted(row["sample_id_sha256"] for row in evaluation)
        ),
    }
    return calibration, evaluation, public


def build_repeated_match_plans(
    members: Sequence[dict], nonmembers: Sequence[dict], *, client_id: int
) -> tuple[list[dict], list[dict]]:
    _validate_pool_pair(members, nonmembers)
    plans = []
    balance_rows = []
    for repeat in range(ATTACK_REPEATS):
        split_seed = ATTACK_SEED + 10_007 * int(client_id) + repeat
        cal_m, eval_m, member_split = source_group_atomic_split(
            members,
            fraction=CALIBRATION_FRACTION,
            seed=split_seed,
            label="member-source-split",
        )
        cal_n, eval_n, nonmember_split = source_group_atomic_split(
            nonmembers,
            fraction=CALIBRATION_FRACTION,
            seed=split_seed,
            label="nonmember-source-split",
        )
        cal_groups = {row["source_group_sha256"] for row in cal_m + cal_n}
        eval_groups = {row["source_group_sha256"] for row in eval_m + eval_n}
        if cal_groups & eval_groups:
            raise RuntimeError("Calibration/evaluation source groups overlap")
        matches = {}
        for index, (partition, part_members, part_nonmembers) in enumerate((
            ("calibration", cal_m, cal_n),
            ("evaluation", eval_m, eval_n),
        )):
            match = coarsened_exact_match(
                part_members,
                part_nonmembers,
                seed=MATCH_SEED + 20_011 * int(client_id) + 2 * repeat + index,
            )
            matches[partition] = match
            balance_rows.append({
                "client_id": int(client_id),
                "repeat": repeat,
                "partition": partition,
                "member_before": match["member_count_before"],
                "nonmember_before": match["nonmember_count_before"],
                "matched_pairs": match["matched_pairs"],
                "retention": match["retention_of_smaller_pool"],
                "pre_max_abs_smd": match["pre_match_balance"]["max_abs_smd"],
                "post_max_abs_smd": match["post_match_balance"]["max_abs_smd"],
                "nonmember_fpr_resolution": 1.0 / match["matched_pairs"],
                "match_plan_sha256": match["match_plan_sha256"],
            })
        public = {
            "repeat": repeat,
            "member_source_split": member_split,
            "nonmember_source_split": nonmember_split,
            "calibration_match_plan_sha256": matches["calibration"][
                "match_plan_sha256"
            ],
            "evaluation_match_plan_sha256": matches["evaluation"][
                "match_plan_sha256"
            ],
            "calibration_pairs": matches["calibration"]["matched_pairs"],
            "evaluation_pairs": matches["evaluation"]["matched_pairs"],
            "calibration_nonmember_fpr_resolution": (
                1.0 / matches["calibration"]["matched_pairs"]
            ),
            "evaluation_nonmember_fpr_resolution": (
                1.0 / matches["evaluation"]["matched_pairs"]
            ),
        }
        public["plan_sha256"] = json_sha256(public)
        plans.append({
            **public,
            "calibration_pairs_data": matches["calibration"]["pairs"],
            "evaluation_pairs_data": matches["evaluation"]["pairs"],
        })
    return plans, balance_rows


def _calibrate_low_fpr(labels, scores, target_fpr: float) -> dict:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    admissible = np.flatnonzero(fpr <= float(target_fpr) + 1e-12)
    best_tpr = float(np.max(tpr[admissible]))
    candidates = [
        int(index)
        for index in admissible
        if math.isclose(float(tpr[index]), best_tpr, abs_tol=1e-12)
    ]
    finite = [index for index in candidates if np.isfinite(thresholds[index])]
    index = min(
        finite or candidates,
        key=lambda value: (float(fpr[value]), -float(thresholds[value])),
    )
    threshold = float(thresholds[index])
    if not math.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
    return {
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "calibration_tpr": float(tpr[index]),
        "calibration_fpr": float(fpr[index]),
    }


def _apply_threshold(labels, scores, threshold: float) -> dict:
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= float(threshold)
    members, nonmembers = labels == 1, labels == 0
    tp = int(np.sum(predictions & members))
    fp = int(np.sum(predictions & nonmembers))
    member_count, nonmember_count = int(np.sum(members)), int(np.sum(nonmembers))
    return {
        "tpr": float(tp / member_count),
        "achieved_fpr": float(fp / nonmember_count),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "member_count": member_count,
        "nonmember_count": nonmember_count,
        "fpr_resolution": float(1.0 / nonmember_count),
    }


def attack_once(
    member_map: Mapping[str, Mapping],
    nonmember_map: Mapping[str, Mapping],
    calibration_pairs: Sequence[tuple[str, str]],
    evaluation_pairs: Sequence[tuple[str, str]],
    *,
    score_field: str,
) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    calibration_labels = np.asarray(
        [1] * len(calibration_pairs) + [0] * len(calibration_pairs), dtype=np.int64
    )
    calibration_raw = np.asarray(
        [-float(member_map[left][score_field]) for left, _ in calibration_pairs]
        + [-float(nonmember_map[right][score_field]) for _, right in calibration_pairs],
        dtype=np.float64,
    )
    raw_auc = float(roc_auc_score(calibration_labels, calibration_raw))
    direction = 1.0 if raw_auc >= 0.5 else -1.0
    calibration_scores = direction * calibration_raw
    fpr, tpr, thresholds = roc_curve(
        calibration_labels, calibration_scores, drop_intermediate=False
    )
    objective = tpr - fpr
    best = np.flatnonzero(np.isclose(objective, np.max(objective), atol=1e-12))
    finite = [int(index) for index in best if np.isfinite(thresholds[index])]
    threshold = float(thresholds[finite[0] if finite else int(best[0])])
    if not math.isfinite(threshold):
        threshold = float(np.nextafter(np.max(calibration_scores), np.inf))

    evaluation_labels = np.asarray(
        [1] * len(evaluation_pairs) + [0] * len(evaluation_pairs), dtype=np.int64
    )
    evaluation_scores = direction * np.asarray(
        [-float(member_map[left][score_field]) for left, _ in evaluation_pairs]
        + [-float(nonmember_map[right][score_field]) for _, right in evaluation_pairs],
        dtype=np.float64,
    )
    predictions = evaluation_scores >= threshold
    operating_points = {}
    for target in FPR_TARGETS:
        calibrated = _calibrate_low_fpr(
            calibration_labels, calibration_scores, target
        )
        operating_points[f"{int(target * 100)}pct"] = {
            **calibrated,
            **_apply_threshold(
                evaluation_labels, evaluation_scores, calibrated["threshold"]
            ),
        }
    return {
        "direction": (
            "lower_loss_is_more_likely_member"
            if direction > 0
            else "higher_loss_is_more_likely_member"
        ),
        "threshold": threshold,
        "auc_roc": float(roc_auc_score(evaluation_labels, evaluation_scores)),
        "asr": float(np.mean(predictions == evaluation_labels)),
        "operating_points": operating_points,
    }


def repeated_matched_attack(
    member_map: Mapping[str, Mapping],
    nonmember_map: Mapping[str, Mapping],
    plans: Sequence[Mapping],
    *,
    score_field: str,
) -> dict:
    runs = []
    for plan in plans:
        attack = attack_once(
            member_map,
            nonmember_map,
            plan["calibration_pairs_data"],
            plan["evaluation_pairs_data"],
            score_field=score_field,
        )
        runs.append({
            "repeat": int(plan["repeat"]),
            "plan_sha256": plan["plan_sha256"],
            "calibration_pairs": int(plan["calibration_pairs"]),
            "evaluation_pairs": int(plan["evaluation_pairs"]),
            "calibration_nonmember_fpr_resolution": plan[
                "calibration_nonmember_fpr_resolution"
            ],
            "evaluation_nonmember_fpr_resolution": plan[
                "evaluation_nonmember_fpr_resolution"
            ],
            **attack,
        })
    getters = {
        "auc_roc": lambda row: row["auc_roc"],
        "asr": lambda row: row["asr"],
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
    for name, getter in getters.items():
        values = [float(getter(row)) for row in runs]
        summary[name] = {
            "mean": statistics.fmean(values),
            "repeat_sample_sd": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "score": score_field,
        "attack_repeats": ATTACK_REPEATS,
        "repeat_plan_sha256": json_sha256([row["plan_sha256"] for row in runs]),
        "summary": summary,
        "runs": runs,
    }


def _method_seed_macro(client_rows: Sequence[dict]) -> dict:
    output = {}
    for score in SCORES:
        output[score] = {}
        for metric in ATTACK_METRICS:
            values = [
                float(row["attacks"][score]["summary"][metric]["mean"])
                for row in client_rows
            ]
            output[score][metric] = {
                "client_macro_mean": statistics.fmean(values),
                "client_sample_sd": statistics.stdev(values),
            }
    return output


def _aggregate(report_seeds: Mapping[int, dict]) -> tuple[list[dict], list[dict]]:
    if tuple(sorted(int(seed) for seed in report_seeds)) != SEEDS:
        raise ValueError(f"Aggregation requires exactly paired seeds {SEEDS}")
    rows = []
    for method in METHODS:
        for score in SCORES:
            for metric in ATTACK_METRICS:
                values = [
                    float(
                        report_seeds[seed]["methods"][method]["macro"][score][metric][
                            "client_macro_mean"
                        ]
                    )
                    for seed in SEEDS
                ]
                rows.append({
                    "method": method,
                    "score": score,
                    "metric": metric,
                    "n_paired_replicates": len(values),
                    "mean": statistics.fmean(values),
                    "replicate_sample_sd": statistics.stdev(values),
                    "seed_42": values[0],
                    "seed_43": values[1],
                    "seed_44": values[2],
                })
    paired = []
    for baseline in METHODS:
        if baseline == "lora":
            continue
        for score in ("trained_loss", "delta_loss"):
            for metric in ("auc_roc", "asr"):
                raw_differences = []
                chance_excess_differences = []
                for seed in SEEDS:
                    proposal = float(
                        report_seeds[seed]["methods"]["lora"]["macro"][score][metric][
                            "client_macro_mean"
                        ]
                    )
                    reference = float(
                        report_seeds[seed]["methods"][baseline]["macro"][score][metric][
                            "client_macro_mean"
                        ]
                    )
                    raw_differences.append(proposal - reference)
                    chance_excess_differences.append(
                        max(0.0, proposal - 0.5) - max(0.0, reference - 0.5)
                    )
                paired.append({
                    "proposal": "lora",
                    "baseline": baseline,
                    "score": score,
                    "metric": metric,
                    "raw_mean_difference": statistics.fmean(raw_differences),
                    "raw_difference_sample_sd": statistics.stdev(raw_differences),
                    "nonnegative_chance_excess_mean_difference": statistics.fmean(
                        chance_excess_differences
                    ),
                    "nonnegative_chance_excess_difference_sample_sd": statistics.stdev(
                        chance_excess_differences
                    ),
                    "lower_nonnegative_chance_excess_runs": sum(
                        value < 0 for value in chance_excess_differences
                    ),
                    "nonnegative_chance_excess_ties": sum(
                        math.isclose(value, 0.0, abs_tol=1e-12)
                        for value in chance_excess_differences
                    ),
                    "n_paired_replicates": len(raw_differences),
                    "seed_42_nonnegative_chance_excess_difference": (
                        chance_excess_differences[0]
                    ),
                    "seed_43_nonnegative_chance_excess_difference": (
                        chance_excess_differences[1]
                    ),
                    "seed_44_nonnegative_chance_excess_difference": (
                        chance_excess_differences[2]
                    ),
                })
    return rows, paired


def _summary_markdown(rows: Sequence[dict], balance_rows: Sequence[dict]) -> str:
    lookup = {(row["method"], row["score"], row["metric"]): row for row in rows}
    lines = [
        "# Covariate-matched read-only MIA audit (seeds 42/43/44)\n",
        "For each repeat, source groups are split before outcome-blind coarsened "
        "exact matching is performed independently in calibration and evaluation. "
        "Values are mean $\\pm$ sample SD over three paired training/partition-seed "
        "replicates. Clients and 20 attack splits are nested measurements, not "
        "independent replicates. AUC/ASR have a 0.5 chance baseline; below-chance "
        "values indicate attack instability and are not ranked as extra privacy. "
        "This exploratory post-hoc analysis was motivated by the v1 initial-loss "
        "negative control; it is not preregistered confirmatory evidence or a formal "
        "privacy guarantee.\n",
        "| Method | Score | AUC | Balanced attack accuracy (ASR) | Eval TPR/FPR at cal-target 5% | Eval TPR/FPR at cal-target 10% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for score in SCORES:
            auc = lookup[(method, score, "auc_roc")]
            asr = lookup[(method, score, "asr")]
            tpr = lookup[(
                method, score, "tpr_at_calibration_target_5fpr_threshold"
            )]
            fpr = lookup[(
                method, score,
                "evaluation_fpr_at_calibration_target_5fpr_threshold",
            )]
            tpr10 = lookup[(
                method, score, "tpr_at_calibration_target_10fpr_threshold"
            )]
            fpr10 = lookup[(
                method, score,
                "evaluation_fpr_at_calibration_target_10fpr_threshold",
            )]
            lines.append(
                f"| {method} | {score} | {auc['mean']:.4f} ± "
                f"{auc['replicate_sample_sd']:.4f} | {asr['mean']:.4f} ± "
                f"{asr['replicate_sample_sd']:.4f} | {tpr['mean']:.4f} ± "
                f"{tpr['replicate_sample_sd']:.4f} / {fpr['mean']:.4f} ± "
                f"{fpr['replicate_sample_sd']:.4f} | {tpr10['mean']:.4f} ± "
                f"{tpr10['replicate_sample_sd']:.4f} / {fpr10['mean']:.4f} ± "
                f"{fpr10['replicate_sample_sd']:.4f} |"
            )
    lines.extend([
        "\n## Matching validity\n",
        "Each range below is over 20 repeated source-atomic splits. `FPR resolution` "
        "is $1/n$ for the matched nonmember partition.\n",
        "| Seed | Client | Cal/eval matched-pair minimum | Retention minimum | "
        "Post-match max abs. SMD | Worst FPR resolution |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        for client_id in range(3):
            selected = [
                row
                for row in balance_rows
                if int(row["seed"]) == seed and int(row["client_id"]) == client_id
            ]
            by_partition = {
                partition: [row for row in selected if row["partition"] == partition]
                for partition in ("calibration", "evaluation")
            }
            lines.append(
                f"| {seed} | {client_id} | "
                f"{min(row['matched_pairs'] for row in by_partition['calibration'])}/"
                f"{min(row['matched_pairs'] for row in by_partition['evaluation'])} | "
                f"{min(row['retention'] for row in selected):.4f} | "
                f"{max(row['post_max_abs_smd'] for row in selected):.4f} | "
                f"{max(row['nonmember_fpr_resolution'] for row in selected):.4f} |"
            )
    lines.extend([
        "\n## Interpretation boundary\n",
        "- Matching controls background, capped per-class instance composition, "
        "object count, and mean normalized box-area bins. Image size and aspect ratio "
        "are additionally audited by SMD; unobserved scene/sensor/source confounding "
        "can remain.",
        "- `trained_loss` is the primary label-aware endpoint attack. `initial_loss` "
        "is a negative control and `delta_loss` is a non-standard sensitivity diagnostic.",
        "- Score direction, Youden-J threshold, and low-FPR thresholds are selected "
        "on calibration data only and frozen for evaluation.",
        "- The 1% operating point is intentionally omitted here because the matched "
        "local partitions cannot consistently resolve 1% FPR. The v1 unmatched audit "
        "retains its explicitly achieved 1% FPR diagnostic.",
        "- This is not a black-box confidence/entropy attack, server update-channel "
        "attack, LiRA, differential privacy, or a confidentiality proof.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve(strict=True)
    args.output_dir = str(
        Path(args.output_dir).resolve(strict=False)
        if args.output_dir
        else root / OUTPUT_NAME
    )
    return args


def _output_snapshot(staging: Path, final: Path, names: Sequence[str]) -> dict:
    output = {}
    for name in names:
        source = staging / name
        destination = final / name
        output[str(destination)] = {
            "path": str(destination),
            "size_bytes": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    return output


def run(argv=None) -> int:
    os.umask(0o077)
    args = _parse_args(argv)
    results_root = Path(args.results_root).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve(strict=False)
    if not args.dry_run:
        validate_output_location(results_root, output_dir)
    elif output_dir != results_root / OUTPUT_NAME:
        raise ValueError(
            f"Phase-2 dry-run also requires canonical output {results_root / OUTPUT_NAME}"
        )

    try:
        from scripts.summarize_mia_robustness import load_reports
    except ModuleNotFoundError:  # pragma: no cover - direct server CLI
        from summarize_mia_robustness import load_reports

    protected_paths: set[Path] = set()
    for seed in SEEDS:
        audit_dir = results_root / _audit_dir_name(seed)
        report_path = audit_dir / "audit_report.json"
        integrity_path = audit_dir / "integrity_manifest.json"
        protected_paths.update((report_path, integrity_path))
        integrity = _load_json(integrity_path)
        before = integrity.get("before")
        if not isinstance(before, dict) or not before:
            raise ValueError(f"Missing protected v1 snapshot: {integrity_path}")
        protected_paths.update(Path(path) for path in before)
        for method in METHODS:
            for client_id in range(3):
                for membership in ("member", "nonmember"):
                    protected_paths.update(
                        _cache_path(audit_dir, method, client_id, membership)
                    )
    protected_before = _snapshot(protected_paths)
    v1_reports, provenance = load_reports(results_root, SEEDS)

    seed_results: dict[int, dict] = {}
    balance_rows: list[dict] = []
    dry_plan_rows: list[dict] = []
    for seed in SEEDS:
        report = v1_reports[seed]
        audit_dir = results_root / _audit_dir_name(seed)
        loaded: dict[str, dict[int, dict[str, list[dict]]]] = {}
        for method in METHODS:
            loaded[method] = {}
            for client_id in range(3):
                loaded[method][client_id] = {}
                for membership in ("member", "nonmember"):
                    trained_path, initial_path = _cache_path(
                        audit_dir, method, client_id, membership
                    )
                    trained_key = _expected_cache_key(
                        report,
                        method=method,
                        client_id=client_id,
                        membership=membership,
                        initial=False,
                    )
                    initial_key = _expected_cache_key(
                        report,
                        method=method,
                        client_id=client_id,
                        membership=membership,
                        initial=True,
                    )
                    loaded[method][client_id][membership] = _merge_records(
                        _load_cache(trained_path, trained_key),
                        _load_cache(initial_path, initial_key),
                    )

        method_results = {method: {"clients": []} for method in METHODS}
        for client_id in range(3):
            reference_members = loaded["full_ft"][client_id]["member"]
            reference_nonmembers = loaded["full_ft"][client_id]["nonmember"]
            _validate_pool_pair(reference_members, reference_nonmembers)
            reference_metadata = {
                "member": metadata_sha256(reference_members),
                "nonmember": metadata_sha256(reference_nonmembers),
            }
            for method in METHODS:
                observed = {
                    membership: metadata_sha256(loaded[method][client_id][membership])
                    for membership in ("member", "nonmember")
                }
                if observed != reference_metadata:
                    raise ValueError(
                        f"Seed {seed} client {client_id}: method-independent metadata "
                        f"differs for method={method}"
                    )
                _validate_pool_pair(
                    loaded[method][client_id]["member"],
                    loaded[method][client_id]["nonmember"],
                )

            plans, client_balance = build_repeated_match_plans(
                reference_members, reference_nonmembers, client_id=client_id
            )
            for row in client_balance:
                balance_rows.append({"seed": seed, **row})
            plan_bundle_sha = json_sha256([plan["plan_sha256"] for plan in plans])
            dry_plan_rows.append({
                "seed": seed,
                "client_id": client_id,
                "repeat_plan_sha256": plan_bundle_sha,
                "minimum_calibration_pairs": min(
                    plan["calibration_pairs"] for plan in plans
                ),
                "minimum_evaluation_pairs": min(
                    plan["evaluation_pairs"] for plan in plans
                ),
            })
            if args.dry_run:
                continue
            for method in METHODS:
                member_rows = loaded[method][client_id]["member"]
                nonmember_rows = loaded[method][client_id]["nonmember"]
                member_map = {row["sample_id_sha256"]: row for row in member_rows}
                nonmember_map = {row["sample_id_sha256"]: row for row in nonmember_rows}
                attacks = {
                    score: repeated_matched_attack(
                        member_map, nonmember_map, plans, score_field=score
                    )
                    for score in SCORES
                }
                digests = {attack["repeat_plan_sha256"] for attack in attacks.values()}
                if digests != {plan_bundle_sha}:
                    raise RuntimeError("Attack scores did not reuse the frozen plan bundle")
                method_results[method]["clients"].append({
                    "client_id": client_id,
                    "sample_metadata_sha256": reference_metadata,
                    "repeat_plan_sha256": plan_bundle_sha,
                    "attacks": attacks,
                })

        if not args.dry_run:
            for method in METHODS:
                if len(method_results[method]["clients"]) != 3:
                    raise RuntimeError(f"Missing client attack results for {method}")
                method_results[method]["macro"] = _method_seed_macro(
                    method_results[method]["clients"]
                )
            seed_results[seed] = {
                "v1_report_sha256": sha256_file(audit_dir / "audit_report.json"),
                "methods": method_results,
            }

    protected_after = _snapshot(protected_paths)
    changes = _changed(protected_before, protected_after)
    if changes:
        raise RuntimeError(f"Protected primary/v1 inputs changed: {changes}")
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_pass",
            "protected_input_files": len(protected_before),
            "matching_protocol": {
                "split_before_match": True,
                "minimum_pairs_per_partition": MIN_MATCHED_PAIRS_PER_PARTITION,
                "minimum_retention_per_partition": MIN_RETENTION_PER_PARTITION,
                "maximum_post_match_absolute_smd": MAX_ABS_SMD,
                "reported_fpr_targets": list(FPR_TARGETS),
            },
            "plans": dry_plan_rows,
            "would_write": str(output_dir),
        }, indent=2, ensure_ascii=False, allow_nan=False))
        print("[PASS] Dry run validated all provenance, source splits, and matching gates")
        print("[PASS] No phase-2 output was written and no attack score was evaluated")
        return 0

    rows, paired_rows = _aggregate(seed_results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "audit_name": OUTPUT_NAME,
        "read_only_primary_and_v1_gate": "pass",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "protocol": {
            "input": "frozen security_audit_v1 own-client loss caches",
            "design_status": (
                "exploratory_post_hoc_sensitivity_analysis_motivated_by_v1_"
                "initial_loss_negative_control"
            ),
            "operation_order": "source_group_atomic_split_then_partition_local_matching",
            "matching": "outcome-blind coarsened exact matching",
            "matching_covariates": [
                "background",
                "per-class instance counts capped at 3",
                "object count capped at 6",
                "log2 mean normalized bbox-area bin",
            ],
            "balance_only_covariates": ["log image pixel area", "log aspect ratio"],
            "match_seed": MATCH_SEED,
            "attack_seed": ATTACK_SEED,
            "attack_repeats": ATTACK_REPEATS,
            "calibration_fraction": CALIBRATION_FRACTION,
            "minimum_matched_pairs_per_partition": MIN_MATCHED_PAIRS_PER_PARTITION,
            "minimum_retention_per_partition": MIN_RETENTION_PER_PARTITION,
            "maximum_post_match_absolute_smd": MAX_ABS_SMD,
            "reported_fpr_targets": list(FPR_TARGETS),
            "one_percent_fpr_omitted_for_resolution": True,
            "paired_difference_chance_excess_definition": (
                "max(0, metric - 0.5); a descriptive below-chance-safe transform, "
                "not the conventional MIA advantage TPR-minus-FPR"
            ),
        },
        "input_provenance": provenance,
        "seed_results": seed_results,
        "aggregate_rows": rows,
        "paired_differences": paired_rows,
        "balance": balance_rows,
        "limitations": [
            "Observed-covariate matching cannot remove unobserved acquisition or scene confounding.",
            "The three paired training/partition seeds are the independent replicate units.",
            "Repeated attack splits quantify split sensitivity, not confidence intervals.",
            "This is a label-aware endpoint loss attack, not a black-box or server-update attack.",
            "Below-chance AUC/ASR is attack instability, not evidence of extra privacy.",
            "The matching protocol was designed post hoc after inspecting the v1 "
            "initial-loss negative control and is exploratory rather than confirmatory.",
        ],
    }

    # Publish the result bundle atomically. An interrupted staging directory is
    # never mistaken for a complete immutable audit.
    staging = output_dir.with_name(f".{OUTPUT_NAME}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(mode=0o700)
    output_names = (
        "audit_report.json",
        "summary_long.csv",
        "paired_differences.csv",
        "matching_balance.csv",
        "summary.md",
    )
    _atomic_json(report, staging / "audit_report.json")
    _atomic_csv(rows, tuple(rows[0]), staging / "summary_long.csv")
    _atomic_csv(paired_rows, tuple(paired_rows[0]), staging / "paired_differences.csv")
    _atomic_csv(balance_rows, tuple(balance_rows[0]), staging / "matching_balance.csv")
    _atomic_text(_summary_markdown(rows, balance_rows), staging / "summary.md")
    protected_publish = _snapshot(protected_paths)
    publish_changes = _changed(protected_before, protected_publish)
    if publish_changes:
        raise RuntimeError(
            f"Protected inputs changed while staging the output: {publish_changes}"
        )
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "read_only_primary_and_v1_gate": True,
        "before": protected_before,
        "after": protected_publish,
        "changes": [],
        "outputs": _output_snapshot(staging, output_dir, output_names),
    }
    _atomic_json(integrity, staging / "integrity_manifest.json")
    _load_json(staging / "audit_report.json")
    _load_json(staging / "integrity_manifest.json")
    os.replace(staging, output_dir)
    print("[PASS] Split-first covariate matching and three-seed aggregation completed")
    print(f"[PASS] {len(protected_before)} primary/v1 inputs are byte-identical")
    print(f"[Results] {output_dir / 'summary.md'}")
    return 0


def main(argv=None) -> int:
    try:
        return run(argv)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
