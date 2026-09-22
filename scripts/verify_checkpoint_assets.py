"""Read-only verification of validation-selected checkpoint release assets.

Run on the machine that holds the historical experiment directory. The verifier
does not load PyTorch checkpoints, create files, or modify model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional


DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoint_index.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_index(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if not isinstance(index, dict) or index.get("schema_version") != 1:
        raise ValueError("expected checkpoint index schema_version=1")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint index must contain a nonempty records list")
    if index.get("record_count") != len(records):
        raise ValueError("record_count does not match the records list")

    names, paths = set(), set()
    total_bytes = 0
    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"record {number} is not an object")
        relative = record.get("historical_project_relative_path")
        asset = record.get("release_asset_name")
        size = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"record {number} has no relative path")
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
            raise ValueError(f"record {number} has an unsafe relative path: {relative}")
        if "\\" in relative or "//" in relative:
            raise ValueError(f"record {number} has a noncanonical relative path: {relative}")
        if not isinstance(asset, str) or not asset or Path(asset).name != asset:
            raise ValueError(f"record {number} has an invalid release asset name")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"record {number} has an invalid byte size")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"record {number} has an invalid SHA-256")
        if asset in names or relative in paths:
            raise ValueError(f"record {number} duplicates an asset name or path")
        names.add(asset)
        paths.add(relative)
        total_bytes += size
    if index.get("total_bytes") != total_bytes:
        raise ValueError("total_bytes does not match the records list")
    return records


def verify_records(project_dir: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stream each file once and return a status for each indexed checkpoint."""
    project_root = project_dir.resolve()
    results = []
    for record in records:
        relative = record["historical_project_relative_path"]
        candidate = project_root.joinpath(*PurePosixPath(relative).parts)
        resolved = candidate.resolve()
        result = {
            "release_asset_name": record["release_asset_name"],
            "path": relative,
            "expected_bytes": record["bytes"],
            "expected_sha256": record["sha256"],
            "status": "ok",
        }
        try:
            resolved.relative_to(project_root)
        except ValueError:
            result["status"] = "path_outside_project"
        else:
            if not candidate.is_file():
                result["status"] = "missing"
            else:
                actual_size = candidate.stat().st_size
                result["actual_bytes"] = actual_size
                if actual_size != record["bytes"]:
                    result["status"] = "size_mismatch"
                else:
                    digest = hashlib.sha256()
                    with candidate.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            digest.update(chunk)
                    result["actual_sha256"] = digest.hexdigest()
                    if result["actual_sha256"] != record["sha256"]:
                        result["status"] = "sha256_mismatch"
        results.append(result)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = sum(result["status"] == "ok" for result in results)
    return {
        "status": "pass" if passed == len(results) else "fail",
        "checked": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "verified_bytes": sum(
            result["expected_bytes"] for result in results if result["status"] == "ok"
        ),
        "results": results,
    }


def emit(summary: Dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif output_format == "tsv":
        print("status\trelease_asset_name\tpath\texpected_bytes\texpected_sha256")
        for row in summary["results"]:
            print(
                "\t".join(
                    str(row[key])
                    for key in (
                        "status",
                        "release_asset_name",
                        "path",
                        "expected_bytes",
                        "expected_sha256",
                    )
                )
            )
    else:
        print(
            f"[{summary['status'].upper()}] checkpoints "
            f"{summary['passed']}/{summary['checked']} verified; "
            f"bytes={summary['verified_bytes']}"
        )
        for row in summary["results"]:
            if row["status"] != "ok":
                print(f"  {row['status']}: {row['path']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--format", choices=("text", "tsv", "json"), default="text")
    args = parser.parse_args(argv)
    if not args.project_dir.is_dir():
        parser.error(f"project directory does not exist: {args.project_dir}")
    try:
        records = load_index(args.index)
        results = verify_records(args.project_dir, records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[INVALID] {exc}", file=sys.stderr)
        return 2
    summary = summarize(results)
    emit(summary, args.format)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
