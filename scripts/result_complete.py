#!/usr/bin/env python3
"""Exit successfully only for a complete, parseable experiment result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys


_PATH_COMPONENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: dict, path: str):
    """Read a dot-delimited mapping path without evaluating user input."""
    components = path.split(".")
    if not components or any(not _PATH_COMPONENT.fullmatch(part) for part in components):
        raise ValueError(f"Unsafe or invalid expectation path: {path!r}")
    current = payload
    for component in components:
        if not isinstance(current, dict) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current


def _parse_expected(specification: str):
    if "=" not in specification:
        raise ValueError(
            f"--expect must use path=value syntax, received {specification!r}"
        )
    path, raw_value = specification.split("=", 1)
    if not path or not raw_value:
        raise ValueError(
            f"--expect must have a non-empty path and value: {specification!r}"
        )
    try:
        expected = json.loads(raw_value)
    except json.JSONDecodeError:
        expected = raw_value
    return path, expected


def _values_equal(actual, expected) -> bool:
    # Do not let Python's bool-is-an-int relationship make true equal to 1.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return (
            math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == expected


def _load_evaluation_arguments(result_path: str) -> dict:
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(result_path)),
                                 "evaluation_manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return {}
    arguments = manifest.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _load_json_object(path: str, label: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root is not an object: {path}")
    return payload


def _validate_completion_manifests(
    result_path: str, *, require_mia: bool
) -> dict:
    """Reject the result/manifest atomic-write window and return train arguments."""
    experiment_dir = os.path.dirname(os.path.abspath(result_path))
    run_path = os.path.join(experiment_dir, "run_manifest.json")
    run_manifest = _load_json_object(run_path, "Run manifest")
    if run_manifest.get("status") != "complete":
        raise ValueError("run_manifest.json has no status='complete' marker")
    training_arguments = run_manifest.get("arguments")
    if not isinstance(training_arguments, dict):
        raise ValueError("run_manifest.json has no arguments object")

    if require_mia:
        evaluation_path = os.path.join(experiment_dir, "evaluation_manifest.json")
        if os.path.isfile(evaluation_path):
            evaluation_manifest = _load_json_object(
                evaluation_path, "Evaluation manifest"
            )
            if evaluation_manifest.get("status") != "complete":
                raise ValueError(
                    "evaluation_manifest.json has no status='complete' marker"
                )
            evaluation_arguments = evaluation_manifest.get("arguments")
            if not isinstance(evaluation_arguments, dict):
                raise ValueError(
                    "evaluation_manifest.json has no arguments object"
                )
            if evaluation_arguments.get("run_mia") is not True:
                raise ValueError(
                    "Completed evaluation manifest was not a MIA evaluation"
                )
        elif training_arguments.get("run_mia") is not True:
            raise ValueError(
                "MIA result has neither a completed MIA evaluation manifest nor "
                "a completed MIA-enabled training manifest"
            )
    return training_arguments


def _require_nonempty_file(path: str, label: str) -> None:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise ValueError(f"Missing or empty {label}: {path}")


def _require_detection_images(
    directory: str, label: str, expected_count: int
) -> None:
    supported = {".jpg", ".jpeg", ".png"}
    if not os.path.isdir(directory):
        raise ValueError(f"Missing {label} directory: {directory}")
    images = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            path = os.path.join(root, filename)
            if os.path.splitext(filename)[1].lower() not in supported:
                continue
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise ValueError(f"Empty or invalid image in {label}: {path}")
            images.append(os.path.relpath(path, directory))
    if len(images) != expected_count:
        raise ValueError(
            f"{label} has {len(images)} images, expected exactly {expected_count}: "
            f"{directory}"
        )
    if len(set(images)) != len(images):
        raise ValueError(f"{label} contains duplicate relative image paths")


def _validation_image_count(
    payload: dict, *, mode: str, client_id: int | None = None
) -> int:
    metadata = payload.get("split_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Result has no embedded split metadata for artifact validation")
    if mode == "centralized":
        split_counts = metadata.get("split_counts", {})
        validation_counts = (
            split_counts.get("val", {}) if isinstance(split_counts, dict) else {}
        )
        value = (
            validation_counts.get("images")
            if isinstance(validation_counts, dict) else None
        )
    else:
        realized = metadata.get("realized_partition_statistics", {})
        records = realized.get("val") if isinstance(realized, dict) else None
        if not isinstance(records, list) or client_id is None:
            raise ValueError("Split metadata has no client validation statistics")
        match = next(
            (
                record for record in records
                if isinstance(record, dict)
                and int(record.get("client_id", -1)) == int(client_id)
            ),
            None,
        )
        value = match.get("num_images") if isinstance(match, dict) else None
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Split metadata has invalid validation image count") from error
    if count <= 0:
        raise ValueError("Split metadata records no validation images")
    return count


def _validate_required_artifacts(
    result_path: str, payload: dict, training_arguments: dict
) -> None:
    """Verify checkpoints and plots promised by a completed study run."""
    experiment_dir = os.path.dirname(os.path.abspath(result_path))
    _require_nonempty_file(
        os.path.join(experiment_dir, "client_data_distribution.png"),
        "client distribution plot",
    )
    mode = payload.get("mode")
    if training_arguments.get("cross_client_eval") is True:
        if mode == "fl":
            matrix = payload.get("cross_client_matrix")
            if not isinstance(matrix, dict):
                raise ValueError(
                    "cross_client_eval=true but FL cross_client_matrix is missing"
                )
        elif mode == "solo":
            records = payload.get("cross_client_test")
            expected_clients = int(payload.get("num_clients", 0))
            if (
                not isinstance(records, list)
                or len(records) != expected_clients
                or not isinstance(payload.get("cross_client_summary"), dict)
            ):
                raise ValueError(
                    "cross_client_eval=true but Solo cross-client evaluation is incomplete"
                )
    if mode == "fl":
        required = (
            ("weights/best_federated.pt", "best federated checkpoint"),
            ("weights/last_federated.pt", "last federated checkpoint"),
            ("fl_training_curves.png", "FL metric plot"),
            ("fl_loss_curves.png", "FL loss plot"),
            ("fl_loss_components.png", "FL loss-component plot"),
            ("fl_learning_rate_schedule.png", "FL learning-rate plot"),
        )
    else:
        required = (
            ("weights/best_full.pt", "best standalone checkpoint"),
            ("weights/last_full.pt", "last standalone checkpoint"),
            ("training_loss_components.png", "standalone loss-component plot"),
            ("learning_rate_schedule.png", "standalone learning-rate plot"),
        )
    for relative_path, label in required:
        _require_nonempty_file(os.path.join(experiment_dir, relative_path), label)

    try:
        interval = int(training_arguments.get("visualize_interval", 5))
        vis_samples = int(training_arguments.get("vis_samples", 6))
    except (TypeError, ValueError) as error:
        raise ValueError("Run manifest has invalid visualization settings") from error
    if interval <= 0:
        return
    if vis_samples <= 0:
        raise ValueError("Run manifest has non-positive vis_samples")

    if mode == "fl":
        rounds = payload.get("round_metrics", [])
        if not isinstance(rounds, list):
            raise ValueError("FL result round_metrics is not a list")
        scheduled_rounds = []
        for record in rounds:
            if not isinstance(record, dict):
                continue
            try:
                round_number = int(record.get("round"))
            except (TypeError, ValueError):
                continue
            if round_number > 0 and round_number % interval == 0:
                scheduled_rounds.append(round_number)
        for round_number in scheduled_rounds:
            for client_id in range(int(payload.get("num_clients", 0))):
                expected_count = min(
                    vis_samples,
                    _validation_image_count(
                        payload, mode="fl", client_id=client_id
                    ),
                )
                directory = os.path.join(
                    experiment_dir, "detection_vis", f"round_{round_number:03d}",
                    f"client_{client_id}", "val",
                )
                _require_detection_images(
                    directory,
                    f"round {round_number} client {client_id} detection visualization",
                    expected_count,
                )
    else:
        history = payload.get("training", {}).get("history", [])
        if not isinstance(history, list):
            raise ValueError("Standalone result training history is not a list")
        scheduled_epochs = []
        for record in history:
            if not isinstance(record, dict) or "val" not in record:
                continue
            try:
                epoch = int(record.get("epoch"))
            except (TypeError, ValueError):
                continue
            if epoch > 0 and epoch % interval == 0:
                scheduled_epochs.append(epoch)
        for epoch in scheduled_epochs:
            client_id = (
                int(payload.get("client_id", -1)) if mode == "solo" else None
            )
            expected_count = min(
                vis_samples,
                _validation_image_count(
                    payload, mode=mode, client_id=client_id
                ),
            )
            directory = os.path.join(
                experiment_dir, "detection_vis", f"epoch_{epoch:03d}", "val"
            )
            _require_detection_images(
                directory, f"epoch {epoch} detection visualization", expected_count
            )


def _mia_setting(payload: dict, result_path: str, key: str):
    """Resolve the attack-time setting, preferring the current evaluation run."""
    candidate_paths = (
        f"experiment.{key}",
        f"training_experiment.{key}",
        f"mia.protocol.{key}",
        f"mia.{key}",
    )
    for path in candidate_paths:
        try:
            value = _nested(payload, path)
        except KeyError:
            continue
        if value is not None:
            return value

    evaluation_arguments = _load_evaluation_arguments(result_path)
    if evaluation_arguments.get(key) is not None:
        return evaluation_arguments[key]

    # Older result schemas record the requested calibration fraction in each
    # attack record even though they do not record mia_max_samples explicitly.
    if key == "mia_calibration_fraction":
        mia = payload.get("mia", {})
        records = mia.get("per_client", []) if isinstance(mia, dict) else []
        observed = set()
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            metrics = record.get("metrics", record)
            if isinstance(metrics, dict) and metrics.get("requested_calibration_fraction") is not None:
                observed.add(float(metrics["requested_calibration_fraction"]))
        if len(observed) == 1:
            return observed.pop()
    raise KeyError(key)


def _validate_mia_block(payload: dict) -> None:
    mia = payload.get("mia")
    if not isinstance(mia, dict):
        raise ValueError("Result has no completed MIA block")
    per_client = mia.get("per_client")
    if not isinstance(per_client, list) or not per_client:
        raise ValueError("MIA block has no per-client attack records")
    configuration = mia.get("configuration")
    expected_source_policy = (
        "exclude_test_source_components_present_in_any_train_client"
    )
    if not isinstance(configuration, dict) or configuration.get(
        "nonmember_source_policy"
    ) != expected_source_policy:
        raise ValueError(
            "MIA block does not record the required source-disjoint nonmember policy"
        )
    expected_clients = 1 if payload.get("mode") == "solo" else int(payload.get("num_clients", 0))
    if expected_clients <= 0 or len(per_client) != expected_clients:
        raise ValueError(
            f"MIA per-client record count is {len(per_client)}, expected {expected_clients}"
        )
    for index, entry in enumerate(per_client):
        metrics = entry.get("metrics", entry) if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            raise ValueError(f"MIA per-client record {index} is not a metric object")
        source_audit = metrics.get("source_disjoint_sampling")
        if not (
            isinstance(source_audit, dict)
            and source_audit.get("policy") == expected_source_policy
            and source_audit.get("member_nonmember_source_group_intersection") == 0
        ):
            raise ValueError(
                f"MIA per-client record {index} is not source-disjoint"
            )
        for metric in ("auc_roc", "tpr_at_1fpr", "asr"):
            value = metrics.get(metric)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"MIA per-client record {index} has invalid {metric}")
    macro = mia.get("macro")
    if not isinstance(macro, dict):
        raise ValueError("MIA block has no macro summary")
    for metric in ("auc_roc", "tpr_at_1fpr", "asr"):
        record = macro.get(metric)
        value = record.get("macro_mean") if isinstance(record, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"MIA macro summary is missing numeric {metric}.macro_mean")
        if not math.isfinite(float(value)):
            raise ValueError(f"MIA macro summary has non-finite {metric}.macro_mean")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--require_mia", action="store_true")
    parser.add_argument(
        "--expect", action="append", default=[], metavar="PATH=VALUE",
        help="Require an exact nested result value (JSON value, then raw-string fallback)",
    )
    parser.add_argument(
        "--expect_split_file", metavar="PATH",
        help="Require the result's split digest to match this manifest file",
    )
    parser.add_argument(
        "--expect_file_sha256", action="append", nargs=2, default=[],
        metavar=("RESULT_PATH", "FILE"),
        help="Require a nested result SHA-256 field to match an actual file",
    )
    parser.add_argument("--expect_mia_max_samples", type=int)
    parser.add_argument("--expect_mia_calibration_fraction", type=float)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        raise FileNotFoundError(args.path)
    with open(args.path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Result root is not an object")
    if payload.get("status") != "complete":
        raise ValueError("Result has no status='complete' marker")
    if int(payload.get("result_schema_version", -1)) < 2:
        raise ValueError("Unsupported or missing result schema")
    mode = payload.get("mode")
    required = {
        "solo": ("own_client_test", "common_test", "parameter_counts"),
        "centralized": ("client_summary", "common_test", "parameter_counts"),
        "fl": ("client_summary", "common_test", "communication", "parameter_counts"),
    }
    if mode not in required:
        raise ValueError(f"Unknown result mode: {mode!r}")
    missing = [key for key in required[mode] if key not in payload]
    if missing:
        raise ValueError(f"Incomplete {mode} result; missing {missing}")
    training_arguments = _validate_completion_manifests(
        args.path, require_mia=args.require_mia
    )
    _validate_required_artifacts(args.path, payload, training_arguments)
    if args.require_mia:
        _validate_mia_block(payload)

    for specification in args.expect:
        path, expected = _parse_expected(specification)
        try:
            actual = _nested(payload, path)
        except KeyError as error:
            raise ValueError(f"Result is missing expected field {path!r}") from error
        if not _values_equal(actual, expected):
            raise ValueError(
                f"Result expectation mismatch for {path}: "
                f"actual={actual!r}, expected={expected!r}"
            )

    if args.expect_split_file:
        split_file = os.path.abspath(args.expect_split_file)
        if not os.path.isfile(split_file):
            raise FileNotFoundError(split_file)
        actual_digest = _file_sha256(split_file)
        stored_digest = payload.get("split_manifest_sha256")
        if stored_digest is None:
            stored_digest = payload.get("split_file_sha256")
        if not isinstance(stored_digest, str):
            raise ValueError("Result has no split manifest SHA-256")
        if stored_digest.strip().lower() != actual_digest:
            raise ValueError(
                "Result split digest mismatch: "
                f"actual_file={actual_digest}, result={stored_digest!r}"
            )

    for result_path, file_path in args.expect_file_sha256:
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        try:
            stored_digest = _nested(payload, result_path)
        except KeyError as error:
            raise ValueError(
                f"Result is missing SHA-256 field {result_path!r}"
            ) from error
        actual_digest = _file_sha256(file_path)
        if not isinstance(stored_digest, str) or stored_digest.strip().lower() != actual_digest:
            raise ValueError(
                f"Result file digest mismatch for {result_path}: "
                f"actual_file={actual_digest}, result={stored_digest!r}"
            )

    mia_expectations = {
        "mia_max_samples": args.expect_mia_max_samples,
        "mia_calibration_fraction": args.expect_mia_calibration_fraction,
    }
    for key, expected in mia_expectations.items():
        if expected is None:
            continue
        _validate_mia_block(payload)
        try:
            actual = _mia_setting(payload, args.path, key)
        except KeyError as error:
            raise ValueError(f"MIA result does not record {key}") from error
        if not _values_equal(actual, expected):
            raise ValueError(
                f"MIA setting mismatch for {key}: actual={actual!r}, expected={expected!r}"
            )


if __name__ == "__main__":
    try:
        main()
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"[INVALID] {error}", file=sys.stderr)
        raise SystemExit(1) from None
