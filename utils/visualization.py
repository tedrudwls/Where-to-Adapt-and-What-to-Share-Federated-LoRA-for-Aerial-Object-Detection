"""
Visualization utilities for experiment results.

Generates:
1. Training loss curves (per client, per round)
2. AP/AP50 curves over FL rounds
3. Client data distribution heatmap
4. Performance comparison bar charts
5. Communication cost comparison
6. LoRA rank sensitivity analysis
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Font settings for Korean support
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def plot_training_loss(loss_data: Dict[str, List[float]],
                       save_path: str,
                       title: str = "Training Loss",
                       xlabel: str = "Epoch / Step",
                       ylabel: str = "Loss"):
    """
    Plot training loss curves.

    Args:
        loss_data: {label: [loss_values]} dict
        save_path: output file path
        title: plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, losses in loss_data.items():
        ax.plot(range(1, len(losses) + 1), losses, label=label, linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_fl_metrics(round_metrics: List[dict],
                    save_path: str,
                    title: str = "FL Training Progress"):
    """
    Plot AP and AP50 curves over FL rounds.

    Args:
        round_metrics: list of round result dicts
        save_path: output file path
    """
    if not round_metrics:
        raise ValueError("round_metrics is empty")
    rounds = [m["round"] for m in round_metrics]
    avg_aps = [m["avg_AP"] for m in round_metrics]
    avg_ap50s = [m["avg_AP50"] for m in round_metrics]
    std_aps = [m["std_AP"] for m in round_metrics]
    std_ap50s = [m["std_AP50"] for m in round_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AP plot
    ax1.plot(rounds, avg_aps, "b-o", label="Client-macro AP", linewidth=2, markersize=4)
    ax1.fill_between(rounds,
                     [a - s for a, s in zip(avg_aps, std_aps)],
                     [a + s for a, s in zip(avg_aps, std_aps)],
                     alpha=0.2, color="blue")
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("AP (IoU=0.50:0.95)", fontsize=12)
    ax1.set_title("AP over FL Rounds (band = client sample SD)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # AP50 plot
    ax2.plot(rounds, avg_ap50s, "r-o", label="Client-macro AP50", linewidth=2, markersize=4)
    ax2.fill_between(rounds,
                     [a - s for a, s in zip(avg_ap50s, std_ap50s)],
                     [a + s for a, s in zip(avg_ap50s, std_ap50s)],
                     alpha=0.2, color="red")
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("AP50 (IoU=0.50)", fontsize=12)
    ax2.set_title("AP50 over FL Rounds (band = client sample SD)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_client_data_distribution(client_splits: List[List[int]],
                                  coco: dict,
                                  save_path: str):
    """
    Plot heatmap of class distribution across clients.
    """
    from collections import defaultdict

    # Build image_id -> annotations
    img_ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        img_ann_map[ann["image_id"]].append(ann)

    cat_names = {}
    if "categories" in coco:
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    cat_ids = sorted(cat_names) if cat_names else sorted(
        set(a["category_id"] for a in coco["annotations"])
    )

    num_clients = len(client_splits)
    num_cats = len(cat_ids)

    # Build distribution matrix
    # Last column counts background images so retained negative examples are visible.
    dist_matrix = np.zeros((num_clients, num_cats + 1))
    for c_idx, img_ids in enumerate(client_splits):
        for img_id in img_ids:
            noncrowd = [ann for ann in img_ann_map.get(img_id, []) if not ann.get("iscrowd", 0)]
            if noncrowd:
                for ann in noncrowd:
                    cat_idx = cat_ids.index(ann["category_id"])
                    dist_matrix[c_idx, cat_idx] += 1
            else:
                dist_matrix[c_idx, -1] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(num_cats + 1))
    ax.set_xticklabels([cat_names.get(cid, str(cid)) for cid in cat_ids] + ["background images"],
                       rotation=45, ha="right")
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"Client {i}" for i in range(num_clients)])

    # Add text annotations
    for i in range(num_clients):
        for j in range(num_cats + 1):
            ax.text(j, i, f"{int(dist_matrix[i, j])}",
                    ha="center", va="center", fontsize=10,
                    color="white" if dist_matrix[i, j] > max(dist_matrix.max(), 1) * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Object instances / background images")
    ax.set_title("Realized Client Data Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_performance_comparison(results: Dict[str, Dict],
                                save_path: str,
                                metric: str = "AP50"):
    """
    Bar chart comparing different methods' performance.

    Args:
        results: {method_name: {client_0: val, client_1: val, ..., avg: val}}
        save_path: output file path
        metric: metric name for title
    """
    methods = list(results.keys())
    num_methods = len(methods)

    # Extract per-client and average values
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(num_methods)
    width = 0.15

    # Determine number of clients from first method
    first_method = results[methods[0]]
    client_keys = [k for k in first_method.keys() if k.startswith("client_")]
    num_clients = len(client_keys)

    colors = plt.cm.Set2(np.linspace(0, 1, num_clients + 1))

    for c_idx in range(num_clients):
        key = f"client_{c_idx}"
        vals = [results[m].get(key, 0) for m in methods]
        offset = (c_idx - num_clients / 2) * width
        ax.bar(x + offset, vals, width, label=f"Client {c_idx}",
               color=colors[c_idx], alpha=0.8)

    # Average line
    avg_vals = [results[m].get("avg", 0) for m in methods]
    ax.plot(x, avg_vals, "k--o", label="Average", linewidth=2, markersize=8)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} Comparison Across Methods", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_communication_cost(cost_data: Dict[str, Dict],
                            save_path: str):
    """
    Plot communication cost comparison.

    Args:
        cost_data: {method: {total_comm_mb, per_round_mb, trainable_params_m, comm_params_m}}
    """
    methods = list(cost_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total communication
    vals = [cost_data[m].get("total_comm_mb", 0) for m in methods]
    axes[0].bar(methods, vals, color="steelblue", alpha=0.8)
    axes[0].set_title("Total Communication (MB)", fontsize=12)
    axes[0].set_ylabel("MB")
    scale = max(max(vals, default=0.0), 1e-9)
    for i, v in enumerate(vals):
        axes[0].text(i, v + scale * 0.02, f"{v:.1f}", ha="center", fontsize=9)

    # Trainable parameters
    vals = [cost_data[m].get("trainable_params_m", 0) for m in methods]
    axes[1].bar(methods, vals, color="coral", alpha=0.8)
    axes[1].set_title("Trainable Parameters (M)", fontsize=12)
    axes[1].set_ylabel("Millions")
    scale = max(max(vals, default=0.0), 1e-9)
    for i, v in enumerate(vals):
        axes[1].text(i, v + scale * 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Communication parameters
    vals = [cost_data[m].get("comm_params_m", 0) for m in methods]
    axes[2].bar(methods, vals, color="mediumseagreen", alpha=0.8)
    axes[2].set_title("Communication Parameters (M)", fontsize=12)
    axes[2].set_ylabel("Millions")
    scale = max(max(vals, default=0.0), 1e-9)
    for i, v in enumerate(vals):
        axes[2].text(i, v + scale * 0.02, f"{v:.2f}", ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_rank_sensitivity(rank_results: Dict[int, Dict],
                          save_path: str):
    """
    Plot LoRA rank sensitivity analysis.

    Args:
        rank_results: {rank: {AP: val, AP50: val, params: val}}
    """
    ranks = sorted(rank_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    aps = [rank_results[r].get("AP", 0) for r in ranks]
    ap50s = [rank_results[r].get("AP50", 0) for r in ranks]
    params = [rank_results[r].get("trainable_params_m", 0) for r in ranks]

    # Performance vs rank
    ax1.plot(ranks, aps, "b-o", label="AP", linewidth=2, markersize=6)
    ax1.plot(ranks, ap50s, "r-s", label="AP50", linewidth=2, markersize=6)
    ax1.set_xlabel("LoRA Rank", fontsize=12)
    ax1.set_ylabel("Performance", fontsize=12)
    ax1.set_title("Detection Performance vs LoRA Rank", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ranks)

    # Parameters vs rank
    ax2.bar(range(len(ranks)), params, color="steelblue", alpha=0.8)
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.set_xlabel("LoRA Rank", fontsize=12)
    ax2.set_ylabel("Trainable Parameters (M)", fontsize=12)
    ax2.set_title("Parameter Count vs LoRA Rank", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_mia_results(mia_data: Dict[str, Dict],
                     save_path: str):
    """
    Plot MIA (Membership Inference Attack) results comparison.

    Args:
        mia_data: {method: {auc_roc, tpr_at_1fpr, asr}}
    """
    methods = list(mia_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("auc_roc", "MIA AUC-ROC", "Closer to 0.5 is less distinguishable"),
        ("tpr_at_1fpr", "TPR @ 1% FPR", "Lower is more private"),
        ("asr", "MIA ASR", "Closer to 0.5 is more private"),
    ]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, (key, title, subtitle) in zip(axes, metrics):
        vals = [mia_data[m].get(key, 0) for m in methods]
        bars = ax.bar(methods, vals, color=colors[:len(methods)], alpha=0.8)
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_ylabel(key.upper())
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

        # Add reference line for random (0.5)
        if key in ["auc_roc", "asr"]:
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
            ax.legend(fontsize=8)

        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def _parse_yolo_detection_label(label_path: str, num_classes: int):
    """Read one strict five-field YOLO detection label file.

    Empty files are valid background-image labels. Every non-empty row must be
    ``class cx cy width height`` with a valid class and a box fully contained in
    normalized image coordinates (up to the writer's decimal tolerance).
    """
    rows = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(fields)}"
                )
            try:
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid YOLO detection row"
                ) from error
            coordinates = (cx, cy, width, height)
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    f"{label_path}:{line_number}: class {class_id} outside "
                    f"0..{num_classes - 1}"
                )
            if not all(np.isfinite(value) for value in coordinates):
                raise ValueError(
                    f"{label_path}:{line_number}: non-finite YOLO coordinate"
                )
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{label_path}:{line_number}: width/height must be positive"
                )
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(
                    f"{label_path}:{line_number}: normalized coordinate outside [0,1]"
                )
            tolerance = 1e-7
            if (
                cx - width / 2 < -tolerance
                or cx + width / 2 > 1 + tolerance
                or cy - height / 2 < -tolerance
                or cy + height / 2 > 1 + tolerance
            ):
                raise ValueError(
                    f"{label_path}:{line_number}: decoded box exceeds image bounds"
                )
            rows.append((class_id, cx, cy, width, height))
    return rows


def _label_dir_for_image_dir(image_dir: str) -> Optional[str]:
    """Apply Ultralytics' images-to-labels path convention to a directory."""
    parts = list(Path(os.path.abspath(image_dir)).parts)
    image_components = [
        index for index, component in enumerate(parts)
        if component.lower() == "images"
    ]
    if not image_components:
        return None
    parts[image_components[-1]] = "labels"
    return str(Path(*parts))


def _draw_gt_boxes(img, label_rows, img_w: int, img_h: int, names):
    """Draw validated ground-truth YOLO boxes on an image.

    GT boxes are drawn with green dashed-style rectangles to distinguish
    them from prediction boxes (which ultralytics draws in solid colors).
    """
    import cv2

    for cls_id, cx, cy, bw, bh in label_rows:
        # Convert YOLO normalized xywh to pixel xyxy
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)

        color = (0, 255, 0)  # green for GT
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if isinstance(names, dict):
            class_name = names.get(cls_id, cls_id)
        elif isinstance(names, (list, tuple)) and cls_id < len(names):
            class_name = names[cls_id]
        else:
            class_name = cls_id
        label = f"GT:{class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)


def save_detection_samples(rtdetr_lora, data_yaml: str, args,
                           save_dir: str, tag: str = "",
                           num_samples: int = 20, seed: int = 42,
                           splits=("val",), strict: bool = True):
    """
    Run inference on fixed random samples and save detection result images.

    Saves bbox-annotated images for visual comparison across epochs/rounds.
    Uses a fixed seed so the SAME images are selected every time, enabling
    side-by-side comparison of detection quality over training.

    Args:
        rtdetr_lora: RTDETRLoRA wrapper
        data_yaml: path to dataset YAML
        args: experiment arguments
        save_dir: directory to save images
        tag: subfolder name (e.g., "epoch_005", "round_03")
        num_samples: number of images to save per split
        seed: random seed for reproducible image selection
    """
    import yaml
    import random

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {data_yaml}")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive for detection visualization")

    base_path = str(data_cfg.get("path", ""))
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(os.path.join(os.path.dirname(data_yaml), base_path))
    requested_splits = tuple(splits)
    if not requested_splits:
        raise ValueError("At least one visualization split is required")

    # Get class names for GT label drawing
    names = getattr(rtdetr_lora.model, "names", {})
    num_classes = int(data_cfg.get("nc", len(names)))
    if num_classes <= 0:
        raise ValueError(f"Invalid dataset class count for visualization: {num_classes}")

    completed_splits = set()
    for split_name in requested_splits:
        try:
            if split_name not in ("train", "val", "test"):
                raise ValueError(f"Unsupported visualization split: {split_name}")
            split_value = data_cfg.get(split_name)
            if not isinstance(split_value, str) or not split_value.strip():
                raise ValueError(
                    f"Dataset YAML has no string path for requested split {split_name!r}"
                )
            split_path = split_value if os.path.isabs(split_value) else os.path.join(
                base_path, split_value
            )
            nested_image_dir = os.path.join(split_path, "images")
            img_dir = nested_image_dir if os.path.isdir(nested_image_dir) else split_path
            img_dir = os.path.abspath(img_dir)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(
                    f"Requested {split_name} image directory not found: {img_dir}"
                )

            all_images = sorted(
                os.path.join(directory, filename)
                for directory, _, filenames in os.walk(img_dir)
                for filename in filenames
                if os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png"}
            )
            if not all_images:
                raise FileNotFoundError(
                    f"Requested {split_name} directory has no supported JPG/PNG images: {img_dir}"
                )

            label_dir = _label_dir_for_image_dir(img_dir)
            if label_dir is None or not os.path.isdir(label_dir):
                raise FileNotFoundError(
                    f"Requested {split_name} YOLO label directory not found for {img_dir}: "
                    f"{label_dir!r}"
                )

            # Select fixed random samples (same seed = same images every time).
            rng = random.Random(seed)
            selected = rng.sample(all_images, min(int(num_samples), len(all_images)))
            label_rows = []
            for image_path in selected:
                relative_image = os.path.relpath(image_path, img_dir)
                label_relative = os.path.splitext(relative_image)[0] + ".txt"
                label_path = os.path.join(label_dir, label_relative)
                if not os.path.isfile(label_path):
                    raise FileNotFoundError(
                        f"Selected {split_name} image has no YOLO label file: "
                        f"{image_path} -> {label_path}"
                    )
                label_rows.append(
                    _parse_yolo_detection_label(label_path, num_classes)
                )

            # Save directory: {save_dir}/{tag}/{split_name}/
            out_dir = os.path.join(save_dir, tag, split_name)
            os.makedirs(out_dir, exist_ok=True)

            # Disable fuse() to prevent conflict with LoRA layers
            # (predict() internally calls model.fuse() via AutoBackend)
            ultralytics_wrapper = rtdetr_lora.ultralytics_model
            model = rtdetr_lora.model
            if ultralytics_wrapper.model is not model:
                raise RuntimeError(
                    "RTDETRLoRA and Ultralytics wrappers do not reference the same model"
                )
            was_training = model.training
            first_parameter = next(model.parameters())
            original_device = first_parameter.device
            original_dtype = first_parameter.dtype
            original_parameters = dict(model.named_parameters())
            grad_flags = {
                name: parameter.requires_grad
                for name, parameter in original_parameters.items()
            }
            original_fuse = getattr(model, "fuse", None)
            head = model.model[-1] if hasattr(model, "model") and len(model.model) else None
            cache_sentinel = object()
            decoder_cache = {
                name: getattr(head, name, cache_sentinel)
                for name in ("shapes", "anchors", "valid_mask")
            } if head is not None else {}
            fuse_patched = False

            # Model.predict() caches a predictor/AutoBackend that holds the same
            # model object and its setup-time device. The training loop moves the
            # model back to its original device after visualization, so a cached
            # CUDA predictor could later feed CUDA inputs to that CPU model.
            # Rebuild the backend for every visualization and release it before
            # restoring the training model/device.
            try:
                import torch

                if torch.is_inference_mode_enabled():
                    raise RuntimeError(
                        "Visualization state must be prepared outside torch.inference_mode()"
                    )
                # Predictor setup runs under inference_mode and its AutoBackend
                # performs model.to(device).float().  Allocate that storage here
                # so the live trainable parameters never become inference tensors.
                model.to(device=torch.device(args.device), dtype=torch.float32)
                if original_fuse is not None:
                    model.fuse = lambda *a, **kw: model  # no-op
                    fuse_patched = True
                ultralytics_wrapper.predictor = None
                results = ultralytics_wrapper.predict(
                    source=selected,
                    imgsz=args.img_size,
                    device=args.device,
                    conf=0.25,
                    verbose=False,
                    save=False,
                    half=False,
                    quantize=None,
                )

                import cv2
                if len(results) != len(selected):
                    raise RuntimeError(
                        f"Prediction count {len(results)} != selected images {len(selected)}"
                    )
                saved_count = 0
                for idx, result in enumerate(results):
                    # Plot prediction bboxes
                    plotted = result.plot()  # returns BGR numpy array
                    h, w = plotted.shape[:2]

                    # Overlay ground truth parsed and validated before inference.
                    _draw_gt_boxes(plotted, label_rows[idx], w, h, names)

                    relative_image = os.path.relpath(selected[idx], img_dir)
                    save_path = os.path.join(out_dir, relative_image)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if not cv2.imwrite(save_path, plotted):
                        raise OSError(f"cv2.imwrite failed for {save_path}")
                    if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
                        raise OSError(f"Visualization artifact is missing or empty: {save_path}")
                    saved_count += 1
            finally:
                ultralytics_wrapper.predictor = None
                ultralytics_wrapper.model = model
                if fuse_patched:
                    model.fuse = original_fuse

                # Restore device/dtype before gradient flags.  Inference tensors
                # cannot legally be changed back to requires_grad=True first.
                model.to(device=original_device, dtype=original_dtype)
                restored_parameters = dict(model.named_parameters())
                if tuple(restored_parameters) != tuple(original_parameters):
                    raise RuntimeError(
                        "Ultralytics visualization changed the model parameter keys"
                    )
                replaced = [
                    name for name, parameter in restored_parameters.items()
                    if parameter is not original_parameters[name]
                ]
                if replaced:
                    raise RuntimeError(
                        "Ultralytics visualization replaced trainable Parameter objects: "
                        f"{replaced[:5]}"
                    )
                for name, param in restored_parameters.items():
                    param.requires_grad_(grad_flags[name])
                for name, value in decoder_cache.items():
                    if value is cache_sentinel:
                        if hasattr(head, name):
                            delattr(head, name)
                    else:
                        setattr(head, name, value)
                model.train(was_training)
                rtdetr_lora.enforce_frozen_norm_eval()

            if saved_count != len(selected):
                raise RuntimeError(
                    f"Saved {saved_count}/{len(selected)} requested {split_name} "
                    "visualization artifacts"
                )
            completed_splits.add(split_name)
            print(f"  [Viz] Saved {saved_count} {split_name} detection samples → {out_dir}")
        except Exception as e:
            message = f"Detection visualization failed ({split_name}): {e}"
            if strict:
                raise RuntimeError(message) from e
            print(f"  [Viz] {message}")

    if strict and completed_splits != set(requested_splits):
        missing = sorted(set(requested_splits) - completed_splits)
        raise RuntimeError(f"Detection visualization did not complete requested splits: {missing}")


def save_results_json(results: dict, save_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            value = float(obj)
            return value if np.isfinite(value) else None
        elif isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    temporary = save_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(
            convert(results), f, indent=2, ensure_ascii=False, allow_nan=False
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, save_path)
    print(f"[Results] Saved: {save_path}")
