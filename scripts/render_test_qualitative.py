#!/usr/bin/env python3
"""Reproducible, read-only qualitative comparison on official AOD-4 v6 test images.

Selection is a separate, prediction-blind command. It chooses one drone and
one helicopter example from source components absent in *both* raw train and
validation. Render uses only the indexed, validation-selected FL checkpoints.
Neither command writes a split, a result JSON, or a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "artifacts" / "checkpoint_index.json"
SPLITS = ("train", "val", "test")
SELECTION_POLICY = "source_disjoint_gt_median_target_box_area_v1"
SOURCE_IDENTITY_POLICY = "roboflow_source_key_or_exact_sha256_connected_components"
SUFFIX = re.compile(r"\.rf\.[0-9a-f]{20,64}(?=\.[^./\\]+$)", re.IGNORECASE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMGSZ = 640
CONF = 0.25  # A display threshold, not the threshold used to compute AP.

# All five runs share one immutable split, training/partition seed, rank and
# checkpoint-selection protocol. Order is also the fixed panel order.
METHODS = (
    ("decoder", "fl_lora_r8_a0.4_decoder_only", "lora", False, True),
    ("backbone", "fl_lora_r8_a0.4_backbone_only", "lora", True, False),
    ("ab", "fl_lora_r8_a0.4", "lora", True, True),
    ("a", "fl_fedsa_lora_r8_a0.4", "fedsa_lora", True, True),
    ("b", "fl_fixed_share_b_lora_r8_a0.4", "fixed_share_b_lora", True, True),
)
PANELS = (("GT", "decoder", "backbone", "ab"), ("GT", "ab", "a", "b"))
PANEL_TITLES = {
    "GT": "Ground truth",
    "decoder": "FedLoRA-AB / decoder only",
    "backbone": "FedLoRA-AB / backbone only",
    "ab": "FedLoRA-AB / backbone + decoder",
    "a": "Share-A / local B",
    "b": "Share-B / local A",
}
COLORS = ((35, 115, 240), (45, 170, 65), (235, 105, 35), (150, 70, 190))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def safe_relative_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "//" in name:
        raise ValueError(f"Unsafe image filename: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe image filename: {name!r}")
    return path


def image_path(data_root: Path, split: str, name: str) -> Path:
    relative = safe_relative_name(name)
    base = (data_root / split).resolve()
    resolved = base.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise FileNotFoundError(f"Image is absent or outside {base}: {name}")
    return resolved


def source_key(name: str) -> str:
    return SUFFIX.sub("", name.replace("\\", "/").strip().lower())


def tree_digest_from_inventory_rows(rows: list[list]) -> str:
    digest = hashlib.sha256()
    for _, name, content_sha in sorted(rows, key=lambda row: (str(row[1]), int(row[0]))):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_sha).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _inventory_records(manifest: dict, coco_by_split: dict) -> dict:
    metadata = manifest.get("metadata", {})
    if metadata.get("schema_version") != 7:
        raise ValueError("Expected immutable schema-v7 split manifest")
    if metadata.get("source_split_policy") != "official_aod4_v6":
        raise ValueError("Selection requires the unmodified official AOD-4 v6 split")
    inventory = metadata.get("source_hash_inventory", {})
    if inventory.get("identity_policy") != SOURCE_IDENTITY_POLICY:
        raise ValueError("Unexpected source identity policy")
    digest_body = {
        key: inventory.get(key)
        for key in ("schema_version", "identity_policy", "records", "per_split_image_tree_sha256")
    }
    if inventory.get("inventory_sha256") != canonical_sha256(digest_body):
        raise ValueError("Source inventory self-digest mismatch")
    if set(inventory.get("records", {})) != set(SPLITS):
        raise ValueError("Source inventory must cover train/val/test")
    records = {}
    for split in SPLITS:
        expected = {int(image["id"]): str(image["file_name"]) for image in coco_by_split[split]["images"]}
        if len(expected) != len(coco_by_split[split]["images"]):
            raise ValueError(f"Duplicate COCO image ID in {split}")
        rows = inventory["records"][split]
        observed = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"Malformed source inventory row in {split}")
            image_id, filename, digest = int(row[0]), str(row[1]), str(row[2])
            safe_relative_name(filename)
            if image_id in observed or not SHA256.fullmatch(digest):
                raise ValueError(f"Duplicate image ID or invalid digest in {split}")
            observed[image_id] = {"file_name": filename, "sha256": digest}
        if {key: row["file_name"] for key, row in observed.items()} != expected:
            raise ValueError(f"COCO/inventory image IDs or filenames differ in {split}")
        if tree_digest_from_inventory_rows(rows) != inventory["per_split_image_tree_sha256"][split]:
            raise ValueError(f"Source inventory tree digest mismatch: {split}")
        records[split] = observed
    return records


def source_component_details(records: dict) -> tuple[set[int], dict[int, str]]:
    """Return source-clean test IDs and deterministic connected-component IDs."""
    parent = {}
    source_buckets, hash_buckets = {}, {}
    for split in SPLITS:
        for image_id, row in records[split].items():
            node = (split, int(image_id))
            parent[node] = node
            source_buckets.setdefault(source_key(row["file_name"]), []).append(node)
            hash_buckets.setdefault(row["sha256"], []).append(node)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        parent[find(right)] = find(left)

    for buckets in (source_buckets, hash_buckets):
        for members in buckets.values():
            for member in members[1:]:
                union(members[0], member)
    members_by_root = {}
    for split in SPLITS:
        for image_id, row in records[split].items():
            root = find((split, image_id))
            members_by_root.setdefault(root, []).append([
                split, image_id, row["file_name"],
                source_key(row["file_name"]), row["sha256"],
            ])
    group_by_test_image = {}
    for root, members in members_by_root.items():
        group_id = canonical_sha256(sorted(members, key=lambda row: (row[0], row[2], row[1])))
        for split, image_id, *_ in members:
            if split == "test":
                group_by_test_image[image_id] = group_id
    train_val_roots = {
        find((split, image_id))
        for split in ("train", "val") for image_id in records[split]
    }
    eligible = {
        image_id for image_id in records["test"]
        if find(("test", image_id)) not in train_val_roots
    }
    return eligible, group_by_test_image


def source_disjoint_test_ids(records: dict) -> set[int]:
    """Reproduce transitive Roboflow-key OR exact-SHA component exclusion."""
    return source_component_details(records)[0]


def _client_test_owners(manifest: dict) -> dict[int, int]:
    clients = manifest.get("clients")
    if not isinstance(clients, list) or len(clients) != 3:
        raise ValueError("Expected exactly three manifest clients")
    owners = {}
    for client in clients:
        client_id = int(client["client_id"])
        if client_id not in (0, 1, 2):
            raise ValueError(f"Invalid client ID: {client_id}")
        for image_id in client["splits"]["test"]["image_ids"]:
            image_id = int(image_id)
            if image_id in owners:
                raise ValueError(f"Test image {image_id} belongs to multiple clients")
            owners[image_id] = client_id
    return owners


def select_gt_examples(coco_test: dict, eligible_ids: set[int], owners: dict[int, int],
                       records_test: dict, group_by_test_image: Optional[dict[int, str]] = None) -> list[dict]:
    """Pick median-size positive examples using only annotations, never predictions."""
    categories = {str(row["name"]).lower(): int(row["id"]) for row in coco_test["categories"]}
    if "drone" not in categories or "helicopter" not in categories:
        raise ValueError("AOD-4 target classes are absent")
    images = {int(image["id"]): image for image in coco_test["images"]}
    if len(images) != len(coco_test["images"]) or set(owners) != set(images):
        raise ValueError("Manifest client test assignments do not cover COCO test")
    annotations = {}
    for annotation in coco_test["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    selected, used = [], set()
    for class_name in ("drone", "helicopter"):
        target_id = categories[class_name]
        candidates = []
        for image_id in eligible_ids:
            if image_id in used:
                continue
            image = images[image_id]
            target_areas = [
                float(ann["bbox"][2]) * float(ann["bbox"][3])
                / (float(image["width"]) * float(image["height"]))
                for ann in annotations.get(image_id, [])
                if int(ann["category_id"]) == target_id
                and float(ann["bbox"][2]) > 0 and float(ann["bbox"][3]) > 0
            ]
            if target_areas:
                candidates.append((image_id, max(target_areas)))
        if not candidates:
            raise ValueError(f"No source-disjoint test positive for {class_name}")
        median_area = statistics.median(area for _, area in candidates)
        # Nearest to the class's median per-image maximum target box area;
        # stable ID breaks ties. The criterion is fixed before any inference.
        image_id, target_area = min(
            candidates, key=lambda item: (
                abs(math.log(item[1]) - math.log(median_area)), item[0]
            )
        )
        used.add(image_id)
        image = images[image_id]
        chosen_target_boxes = [
            ann["bbox"] for ann in annotations[image_id]
            if int(ann["category_id"]) == target_id
        ]
        zoom_box = max(chosen_target_boxes, key=lambda bbox: float(bbox[2]) * float(bbox[3]))
        selected.append({
            "target_class": class_name,
            "image_id": image_id,
            "file_name": str(image["file_name"]),
            "image_sha256": records_test[image_id]["sha256"],
            "source_group_id": (
                group_by_test_image[image_id] if group_by_test_image is not None else None
            ),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "client_id": owners[image_id],
            "target_max_normalized_box_area": target_area,
            "candidate_median_normalized_box_area": median_area,
            "positive_candidate_count": len(candidates),
            "zoom_xyxy": gt_zoom_xyxy(zoom_box, int(image["width"]), int(image["height"])),
            "annotations": [
                {
                    "category_id": int(ann["category_id"]),
                    "bbox_xywh": [float(value) for value in ann["bbox"]],
                }
                for ann in annotations.get(image_id, [])
            ],
        })
    return selected


def gt_zoom_xyxy(box_xywh: list, width: int, height: int) -> list[int]:
    """A fixed square crop around a GT target; independent of all predictions."""
    x, y, bw, bh = (float(value) for value in box_xywh)
    if width <= 0 or height <= 0 or bw <= 0 or bh <= 0:
        raise ValueError("Invalid image/GT dimensions for zoom")
    side = min(min(width, height), max(192, int(math.ceil(3 * max(bw, bh)))))
    center_x, center_y = x + bw / 2, y + bh / 2
    x1 = max(0, min(width - side, int(round(center_x - side / 2))))
    y1 = max(0, min(height - side, int(round(center_y - side / 2))))
    return [x1, y1, x1 + side, y1 + side]


def _assert_split_protocol(manifest: dict):
    metadata = manifest["metadata"]
    if (metadata.get("partition"), metadata.get("dirichlet_alpha"),
            metadata.get("seed"), metadata.get("num_clients")) != ("dirichlet", 0.4, 42, 3):
        raise ValueError("Expected seed-42, Dirichlet alpha=0.4, three-client split")
    if metadata.get("class_names") != ["airplane", "bird", "drone", "helicopter"]:
        raise ValueError("Unexpected AOD-4 class order")


def _new_output_dir(path: Path):
    if path.exists():
        raise FileExistsError(f"Output directory already exists; refusing overwrite: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _build_selection(data_root: Path, split_file: Path) -> dict:
    """Construct selection entirely from COCO GT and embedded source hashes."""
    data_root, split_file = data_root.resolve(), split_file.resolve()
    manifest = load_json(split_file)
    _assert_split_protocol(manifest)
    coco_by_split = {}
    for split in SPLITS:
        annotation_path = data_root / split / "_annotations.coco.json"
        expected = manifest["metadata"]["annotation_sha256"][split]
        if file_sha256(annotation_path) != expected:
            raise ValueError(f"COCO annotation SHA mismatch: {split}")
        coco_by_split[split] = load_json(annotation_path)
    records = _inventory_records(manifest, coco_by_split)
    eligible, groups = source_component_details(records)
    owners = _client_test_owners(manifest)
    chosen = select_gt_examples(coco_by_split["test"], eligible, owners,
                                records["test"], groups)
    for item in chosen:
        if file_sha256(image_path(data_root, "test", item["file_name"])) != item["image_sha256"]:
            raise ValueError(f"Selected test image hash mismatch: {item['image_id']}")
    payload = {
        "schema_version": 1,
        "selection_policy": SELECTION_POLICY,
        "prediction_blind": True,
        "source_identity_policy": SOURCE_IDENTITY_POLICY,
        "split_file_sha256": file_sha256(split_file),
        "source_inventory_sha256": manifest["metadata"]["source_hash_inventory"]["inventory_sha256"],
        "source_audit_sha256": manifest["metadata"]["cross_split_source_audit"]["audit_sha256"],
        "annotation_sha256": manifest["metadata"]["annotation_sha256"],
        "data_root_at_selection": str(data_root),
        "eligible_source_disjoint_test_images": len(eligible),
        "images": chosen,
    }
    return payload


def select(args):
    payload = _build_selection(args.data_root, args.split_file)
    _new_output_dir(args.output_dir)
    destination = args.output_dir / "selection.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] Prediction-blind selection: {destination}")
    for item in payload["images"]:
        print(f"  {item['target_class']}: image_id={item['image_id']} client={item['client_id']} "
              f"file={item['file_name']}")


def _indexed_methods(index: dict, results_root: Path, project_dir: Path,
                     split_digest: str, model_weights_digest: str) -> list[dict]:
    records = index.get("records", [])
    selected = []
    for key, experiment, method, backbone, decoder in METHODS:
        experiment_id = f"seed_42/{experiment}"
        matches = [row for row in records if row.get("experiment_id") == experiment_id]
        if len(matches) != 1:
            raise ValueError(f"Exactly one indexed checkpoint required: {experiment_id}")
        row = matches[0]
        if (row.get("mode"), row.get("method"), row.get("partition"),
                row.get("partition_seed"), row.get("training_seed"), row.get("rank")) != (
                    "fl", method, "dirichlet", 42, 42, 8):
            raise ValueError(f"Checkpoint index protocol mismatch: {experiment_id}")
        if row.get("split_manifest_sha256") != split_digest:
            raise ValueError(f"Checkpoint index split hash mismatch: {experiment_id}")
        if row.get("pretrained_sha256") != model_weights_digest:
            raise ValueError(f"Pretrained weights hash mismatch: {experiment_id}")
        path = (project_dir / row["historical_project_relative_path"]).resolve()
        if not path.is_relative_to(project_dir.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Indexed best checkpoint unavailable: {path}")
        if path.name != "best_federated.pt":
            raise ValueError(f"Not a validation-best checkpoint: {path}")
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Indexed checkpoint byte/hash mismatch: {path}")
        result_path = results_root / experiment_id / "fl_results.json"
        result = load_json(result_path)
        if (result.get("status"), result.get("mode"), result.get("fl_method"),
                result.get("seed"), result.get("partition_seed"),
                result.get("split_manifest_sha256")) != (
                    "complete", "fl", method, 42, 42, split_digest):
            raise ValueError(f"Primary result protocol mismatch: {result_path}")
        if (result.get("lora_rank"), result.get("apply_lora_backbone"),
                result.get("apply_lora_decoder")) != (8, backbone, decoder):
            raise ValueError(f"LoRA target/rank mismatch: {result_path}")
        selection = result.get("selection", {})
        if (selection.get("criterion"), selection.get("round")) != (
                "best_macro_client_local_val_AP", row["selected_at"]):
            raise ValueError(f"Result and indexed best round disagree: {result_path}")
        if Path(selection.get("checkpoint", "")).name != "best_federated.pt":
            raise ValueError(f"Result did not select best_federated.pt: {result_path}")
        selected.append({
            "key": key, "experiment_id": experiment_id, "method": method,
            "backbone": backbone, "decoder": decoder,
            "checkpoint_path": path, "checkpoint_sha256": row["sha256"],
            "checkpoint_bytes": row["bytes"], "selected_round": row["selected_at"],
            "result_path": result_path, "result_sha256": file_sha256(result_path),
            "result": result,
        })
    return selected


def _model_args(result: dict, model_weights: Path, device: str) -> SimpleNamespace:
    experiment = result.get("training_experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Primary result has no training_experiment configuration")
    args = dict(experiment)
    args["num_classes"] = 4
    args["model_weights"] = str(model_weights)
    args["device"] = device
    if args.get("model_name") != "rtdetr-l" or args.get("img_size") != IMGSZ:
        raise ValueError("Unexpected RT-DETR architecture or input resolution")
    return SimpleNamespace(**args)


def _extract_boxes(result, expected_image: Path, width: int, height: int) -> list[dict]:
    if Path(result.path).resolve() != expected_image.resolve():
        raise ValueError(f"Prediction/image order mismatch: {result.path} != {expected_image}")
    if tuple(result.orig_shape) != (height, width):
        raise ValueError(f"Prediction original shape mismatch: {result.orig_shape}")
    box_group = result.boxes
    if box_group is None:
        return []
    xyxy = box_group.xyxy.detach().cpu().tolist()
    confidence = box_group.conf.detach().cpu().tolist()
    classes = box_group.cls.detach().cpu().tolist()
    if not (len(xyxy) == len(confidence) == len(classes)):
        raise ValueError("Ultralytics box fields have inconsistent lengths")
    boxes = []
    for rect, conf, cls in zip(xyxy, confidence, classes):
        cls_id = int(cls)
        if len(rect) != 4 or cls != cls_id or cls_id not in range(4):
            raise ValueError("Invalid predicted class or rectangle")
        if not all(math.isfinite(float(value)) for value in (*rect, conf)):
            raise ValueError("Non-finite predicted box")
        x1, y1, x2, y2 = (float(value) for value in rect)
        if x1 < -1e-3 or y1 < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3 or x2 < x1 or y2 < y1:
            raise ValueError("Predicted box exceeds image bounds")
        if conf < CONF - 1e-6 or conf > 1 + 1e-6:
            raise ValueError("Predicted confidence violates the fixed display threshold")
        boxes.append({"class_id": cls_id, "confidence": float(conf),
                      "xyxy": [x1, y1, x2, y2]})
    return boxes


def _predict_methods(methods: list[dict], selection: dict, data_root: Path,
                     split_digest: str, model_weights: Path, device: str) -> dict:
    # Heavy imports are intentionally render-only. Selection and unit tests run
    # without CUDA, torch, Ultralytics or an extracted AOD-4 dataset.
    import gc
    import sys
    import torch

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.rtdetr_lora import RTDETRLoRA  # noqa: F401: imported by _new_client_models
    from trainers.fl_server import _apply_checkpoint, _load_checkpoint, _new_client_models

    client_sizes = [
        int(client["splits"]["train"]["num_images"])
        for client in sorted(selection["manifest_clients"], key=lambda row: int(row["client_id"]))
    ]
    data_info = {
        "class_names": ["airplane", "bird", "drone", "helicopter"],
        "split_manifest_sha256": split_digest,
        "client_sizes": client_sizes,
    }
    predictions = {item["image_id"]: {} for item in selection["images"]}
    for method in methods:
        args = _model_args(method["result"], model_weights, device)
        if (args.fl_method, bool(args.apply_lora_backbone),
                bool(args.apply_lora_decoder)) != (
                    method["method"], method["backbone"], method["decoder"]):
            raise ValueError(f"Checkpoint configuration disagrees for {method['experiment_id']}")
        payload = _load_checkpoint(str(method["checkpoint_path"]))
        if (payload.get("round"), payload.get("selection")) != (
                method["selected_round"], "best_macro_client_local_val_AP"):
            raise ValueError(f"Checkpoint itself is not the indexed best round: {method['experiment_id']}")
        if payload.get("experiment") != method["result"].get("training_experiment"):
            raise ValueError("Checkpoint/result training protocol differs")
        models = _new_client_models(args, data_info)
        _apply_checkpoint(payload, models, args, data_info)
        for item in selection["images"]:
            client_id = int(item["client_id"])
            wrapper = models[client_id].ultralytics_model
            model = models[client_id].model
            model.to(device=torch.device(device), dtype=torch.float32)
            model.eval()
            original_fuse = getattr(model, "fuse", None)
            try:
                if original_fuse is not None:
                    model.fuse = lambda *a, **kw: model  # do not merge LoRA into Conv/Linear
                wrapper.predictor = None
                path = image_path(data_root, "test", item["file_name"])
                results = wrapper.predict(
                    source=str(path), imgsz=IMGSZ, device=device, conf=CONF,
                    verbose=False, save=False, half=False, quantize=None,
                )
                if len(results) != 1:
                    raise RuntimeError("Expected exactly one prediction per image")
                predictions[item["image_id"]][method["key"]] = _extract_boxes(
                    results[0], path, item["width"], item["height"]
                )
            finally:
                wrapper.predictor = None
                if original_fuse is not None:
                    model.fuse = original_fuse
                model.to("cpu")
        del models, payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def _panel(image, annotations: list[dict], predictions: Optional[list[dict]],
           class_names: list[str], category_to_label: dict[int, int], title: str):
    from PIL import ImageDraw, ImageFont

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if predictions is None:
        boxes = [
            {
                "class_id": category_to_label[int(annotation["category_id"])],
                "xyxy": [
                    annotation["bbox_xywh"][0], annotation["bbox_xywh"][1],
                    annotation["bbox_xywh"][0] + annotation["bbox_xywh"][2],
                    annotation["bbox_xywh"][1] + annotation["bbox_xywh"][3],
                ],
                "confidence": None,
            }
            for annotation in annotations
        ]
    else:
        boxes = predictions
    for box in boxes:
        class_id = box["class_id"]
        color = COLORS[class_id]
        coords = [float(value) for value in box["xyxy"]]
        draw.rectangle(coords, outline=color, width=max(2, image.width // 350))
        label = class_names[class_id]
        if box["confidence"] is not None:
            label += f" {box['confidence']:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x, y = max(0, int(coords[0])), max(0, int(coords[1]) - text_height - 5)
        draw.rectangle([x, y, x + text_width + 4, y + text_height + 4], fill=color)
        draw.text((x + 2, y + 2), label, fill="white", font=font)
    return canvas


def _draw_figure(selection: dict, predictions: dict, data_root: Path, output: Path):
    from PIL import Image, ImageDraw, ImageFont

    panel_size, header_height, row_label_height, margin = 512, 34, 34, 8
    width = 4 * panel_size + 5 * margin
    row_height = header_height + row_label_height + panel_size + margin
    height = 2 * row_height + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    class_names = ["airplane", "bird", "drone", "helicopter"]
    category_to_label = {int(key): int(value) for key, value in selection["cat_id_to_label"].items()}
    for row_index, item in enumerate(selection["images"]):
        path = image_path(data_root, "test", item["file_name"])
        with Image.open(path) as opened:
            raw = opened.convert("RGB")
        if raw.size != (item["width"], item["height"]):
            raise ValueError(f"Image dimensions changed after selection: {path}")
        top = margin + row_index * row_height
        draw.text((margin, top + 8),
                  f"{item['target_class']} | test image {item['image_id']} | client {item['client_id']}",
                  fill="black", font=font)
        for column, key in enumerate(PANELS[row_index]):
            x = margin + column * (panel_size + margin)
            y = top + header_height
            draw.text((x + 3, y + 8), PANEL_TITLES[key], fill="black", font=font)
            boxes = None if key == "GT" else predictions[item["image_id"]][key]
            panel = _panel(raw, item["annotations"], boxes,
                           class_names, category_to_label, PANEL_TITLES[key])
            # The same GT-defined crop, saved during the prediction-blind
            # selection phase, is shown as an inset for every method.
            crop = tuple(int(value) for value in item["zoom_xyxy"])
            zoom = panel.crop(crop).resize((190, 190), Image.Resampling.LANCZOS)
            panel = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            panel_draw = ImageDraw.Draw(panel)
            sx, sy = panel_size / raw.width, panel_size / raw.height
            panel_draw.rectangle([crop[0] * sx, crop[1] * sy, crop[2] * sx, crop[3] * sy],
                                 outline="white", width=2)
            inset_x, inset_y = panel_size - 190 - 8, panel_size - 190 - 8
            panel_draw.rectangle([inset_x - 3, inset_y - 3, panel_size - 5, panel_size - 5],
                                 fill="white")
            panel.paste(zoom, (inset_x, inset_y))
            sheet.paste(panel, (x, y + row_label_height))
    sheet.save(output, format="PNG")


def render(args):
    selection_path = args.selection.resolve()
    selection = load_json(selection_path)
    if (selection.get("schema_version"), selection.get("selection_policy"),
            selection.get("prediction_blind")) != (1, SELECTION_POLICY, True):
        raise ValueError("Unsupported or prediction-dependent selection manifest")
    if [item.get("target_class") for item in selection.get("images", [])] != ["drone", "helicopter"]:
        raise ValueError("Expected exactly one drone and one helicopter test image")
    split_digest = file_sha256(args.split_file)
    if split_digest != selection["split_file_sha256"]:
        raise ValueError("Selection and rendering split manifests differ")
    manifest = load_json(args.split_file)
    _assert_split_protocol(manifest)
    if manifest["metadata"]["source_hash_inventory"]["inventory_sha256"] != selection["source_inventory_sha256"]:
        raise ValueError("Selection/source inventory mismatch")
    if manifest["metadata"]["cross_split_source_audit"]["audit_sha256"] != selection["source_audit_sha256"]:
        raise ValueError("Selection/source audit mismatch")
    for split in SPLITS:
        if file_sha256(args.data_root / split / "_annotations.coco.json") != selection["annotation_sha256"][split]:
            raise ValueError(f"COCO annotations changed after selection: {split}")
    for item in selection["images"]:
        path = image_path(args.data_root, "test", item["file_name"])
        if file_sha256(path) != item["image_sha256"]:
            raise ValueError(f"Test image changed after selection: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    # Neither checkpoint loading nor inference may write into experiment dirs.
    results_root = args.results_root.resolve()
    if args.output_dir.resolve().is_relative_to(results_root):
        raise ValueError("Output directory must be outside the entire primary results root")
    recomputed = _build_selection(args.data_root, args.split_file)
    if selection != recomputed:
        raise ValueError("Selection file differs from prediction-blind COCO/source reconstruction")
    index = load_json(args.checkpoint_index)
    if index.get("schema_version") != 1 or index.get("record_count") != len(index.get("records", [])):
        raise ValueError("Checkpoint index schema/count mismatch")
    weights_digest = file_sha256(args.model_weights)
    methods = _indexed_methods(index, args.results_root, args.project_dir,
                               split_digest, weights_digest)
    protected = {str(args.split_file.resolve()): split_digest,
                 str(args.model_weights.resolve()): weights_digest}
    for method in methods:
        protected[str(method["checkpoint_path"])] = method["checkpoint_sha256"]
        protected[str(method["result_path"])] = method["result_sha256"]
    selection_with_metadata = {**selection,
                               "manifest_clients": manifest["clients"],
                               "cat_id_to_label": manifest["metadata"]["cat_id_to_label"]}
    predictions = _predict_methods(methods, selection_with_metadata, args.data_root,
                                   split_digest, args.model_weights, args.device)
    for path_string, expected in protected.items():
        if file_sha256(Path(path_string)) != expected:
            raise RuntimeError(f"Protected primary input changed during rendering: {path_string}")
    _new_output_dir(args.output_dir)
    figure_path = args.output_dir / "qualitative_comparison.png"
    _draw_figure(selection_with_metadata, predictions, args.data_root, figure_path)
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "qualitative_examples_only_not_AP_evidence",
        "source_split": "official_aod4_v6_test",
        "source_disjoint_from_train_and_val": True,
        "selection_file": str(selection_path),
        "selection_file_sha256": file_sha256(selection_path),
        "split_file_sha256": split_digest,
        "pretrained_weights_sha256": weights_digest,
        "prediction_protocol": {
            "model": "RT-DETR-L", "imgsz": IMGSZ, "conf": CONF,
            "half": False, "native_rtdetr_postprocessing": True,
            "display_threshold_only": True,
        },
        "panels": [list(row) for row in PANELS],
        "images": selection["images"],
        "methods": [
            {key: method[key] for key in (
                "key", "experiment_id", "method", "backbone", "decoder",
                "checkpoint_sha256", "selected_round", "result_sha256")}
            for method in methods
        ],
        "predictions": {str(key): value for key, value in predictions.items()},
        "protected_primary_sha256": protected,
        "figure_sha256": file_sha256(figure_path),
    }
    report_path = args.output_dir / "predictions.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] Read-only best-checkpoint qualitative comparison: {figure_path}")
    print(f"[PASS] Prediction and integrity manifest: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select", help="GT-only source-disjoint test selection")
    select_parser.add_argument("--data-root", type=Path, required=True)
    select_parser.add_argument("--split-file", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser = subparsers.add_parser("render", help="render five validation-best FL models")
    render_parser.add_argument("--selection", type=Path, required=True)
    render_parser.add_argument("--data-root", type=Path, required=True)
    render_parser.add_argument("--split-file", type=Path, required=True)
    render_parser.add_argument("--project-dir", type=Path, required=True)
    render_parser.add_argument("--results-root", type=Path, required=True)
    render_parser.add_argument("--checkpoint-index", type=Path, default=DEFAULT_INDEX)
    render_parser.add_argument("--model-weights", type=Path, required=True)
    render_parser.add_argument("--device", default="cuda:0")
    render_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        select(args)
    else:
        render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
