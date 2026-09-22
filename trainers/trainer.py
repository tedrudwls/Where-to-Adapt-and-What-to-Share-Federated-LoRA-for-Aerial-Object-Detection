"""One consistent RT-DETR training/evaluation pipeline for every study method."""

from __future__ import annotations

import contextlib
import io
import math
import os
import random
from collections import defaultdict
from copy import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.models.rtdetr.val import RTDETRDataset

from utils.visualization import save_detection_samples


RTDETR_AUGMENTATION_DEFAULTS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
}


def rtdetr_augmentation_manifest(args) -> dict:
    """Record the pinned Ultralytics transform probabilities used by this study."""
    return {
        "implementation": "ultralytics_8.4.126_RTDETRDataset",
        "initial": dict(RTDETR_AUGMENTATION_DEFAULTS),
        "close_mosaic_effective_epochs_requested": int(args.close_mosaic_epochs),
        "close_mosaic_disables": ["mosaic", "mixup", "cutmix", "copy_paste"],
        "rect": False,
        "cache": False,
        "train_only": True,
        "persistent_dataloader_workers": False,
    }


def print_metrics(label: str, metrics: dict):
    print(
        f"{label}: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, "
        f"AP75={metrics['AP75']:.4f}"
    )
    for class_name, values in metrics.get("per_class", {}).items():
        print(
            f"    {class_name:>15s}: AP={values.get('AP', float('nan')):.4f}, "
            f"AP50={values.get('AP50', float('nan')):.4f}"
        )


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_data_yaml(data_yaml: str) -> dict:
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    for key in ("path", "train", "val", "test", "names", "nc"):
        if key not in data:
            raise ValueError(f"{data_yaml}: missing key {key!r}")
    if len(data["names"]) != int(data["nc"]):
        raise ValueError(f"{data_yaml}: names/nc mismatch")
    data.setdefault("channels", 3)
    if int(data["channels"]) != 3:
        raise ValueError("This RT-DETR experiment currently supports RGB input only")
    return data


def build_rtdetr_dataset(data_yaml: str, split: str, args, augment: bool):
    """Mirror RTDETRTrainer.build_dataset with a pinned Ultralytics API."""
    data = _load_data_yaml(data_yaml)
    image_path = data[split]
    if not os.path.isabs(image_path):
        image_path = os.path.join(data["path"], image_path)
    if not os.path.isdir(image_path):
        raise FileNotFoundError(f"{split} image directory not found: {image_path}")

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.imgsz = args.img_size
    cfg.batch = args.batch_size
    cfg.workers = args.num_workers
    cfg.mode = "train" if augment else "val"
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = False
    cfg.classes = None
    cfg.fraction = 1.0
    cfg.seed = args.seed
    cfg.close_mosaic = args.close_mosaic_epochs
    for name, value in RTDETR_AUGMENTATION_DEFAULTS.items():
        setattr(cfg, name, value)

    dataset = RTDETRDataset(
        img_path=image_path,
        imgsz=args.img_size,
        batch_size=args.batch_size,
        augment=augment,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        prefix=f"{split}: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    # Ultralytics' close_mosaic() needs the transform hyperparameters again.
    # Keep this private copy with the dataset so every client closes the exact
    # same transforms on its effective-epoch schedule.
    dataset._study_hyp = cfg
    return dataset


def build_rtdetr_dataloader(data_yaml: str, split: str, args, augment: bool,
                            batch_size: Optional[int] = None,
                            indices: Optional[List[int]] = None):
    dataset = build_rtdetr_dataset(data_yaml, split, args, augment)
    source = Subset(dataset, indices) if indices is not None else dataset
    generator = torch.Generator()
    seed_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    generator.manual_seed(args.seed + seed_offset)
    workers = max(0, int(args.num_workers))
    loader = DataLoader(
        source,
        batch_size=batch_size or args.batch_size,
        shuffle=bool(augment and indices is None),
        num_workers=workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=dataset.collate_fn,
        worker_init_fn=_seed_worker,
        generator=generator,
        # The study mutates train transforms at close_mosaic. Standard
        # persistent worker copies would not reliably observe that mutation.
        persistent_workers=False,
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError(f"No batches built for {data_yaml} split={split}")
    return dataset, loader


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    moved["img"] = moved["img"].float().div_(255.0)
    return moved


def _unpack_loss(output):
    components = {}
    if isinstance(output, (tuple, list)):
        loss = output[0]
        if len(output) > 1 and isinstance(output[1], dict):
            components = output[1]
    elif isinstance(output, dict):
        loss = sum(output.values())
        components = output
    else:
        loss = output
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(f"Expected scalar RT-DETR loss, got {type(loss)} shape={getattr(loss, 'shape', None)}")
    return loss, components


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


@contextlib.contextmanager
def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        yield
        return
    try:
        with torch.amp.autocast(device_type=device.type, enabled=True):
            yield
    except AttributeError:
        with torch.cuda.amp.autocast(enabled=True):
            yield


def _optimizer_groups(rtdetr_lora, args):
    """AdamW groups with common decay policy and documented section LRs."""
    model = rtdetr_lora.model
    backbone_len = len(model.yaml["backbone"])
    backbone_prefixes = tuple(f"model.{index}" for index in range(backbone_len))
    grouped = defaultdict(list)
    seen = set()
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
    )

    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if full_name.endswith(".lora_A") or full_name.endswith(".lora_B"):
                learning_rate = args.lr
            elif rtdetr_lora._is_task_head_key(full_name):
                learning_rate = args.head_lr
            elif args.fl_method == "full_ft" and any(
                full_name == prefix or full_name.startswith(prefix + ".")
                for prefix in backbone_prefixes
            ):
                learning_rate = args.lr * args.backbone_lr_ratio
            else:
                learning_rate = args.lr
            no_decay = parameter_name == "bias" or isinstance(module, norm_types)
            decay = 0.0 if no_decay else args.weight_decay
            grouped[(float(learning_rate), float(decay))].append(parameter)

    if not grouped:
        raise RuntimeError("No trainable parameters were routed to the optimizer")
    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": decay,
        }
        for (learning_rate, decay), parameters in sorted(grouped.items())
    ]


class WarmupCosine:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps - 1))
        self.min_lr_ratio = float(min_lr_ratio)

    def set_step(self, step: int):
        step = min(max(int(step), 0), self.total_steps - 1)
        if self.warmup_steps and step < self.warmup_steps:
            multiplier = (step + 1) / self.warmup_steps
        else:
            denominator = max(1, self.total_steps - self.warmup_steps - 1)
            progress = (step - self.warmup_steps) / denominator
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            multiplier = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        return [group["lr"] for group in self.optimizer.param_groups]


class FLLocalTrainer:
    """Fail-fast local RT-DETR trainer shared by FL and standalone modes."""

    def __init__(self, rtdetr_lora, data_yaml: str, args, client_id: int = 0,
                 total_epochs: Optional[int] = None):
        self.rtdetr_lora = rtdetr_lora
        self.model = rtdetr_lora.model
        self.args = args
        self.client_id = client_id
        self.device = torch.device(args.device)
        self.data_yaml = data_yaml
        self.total_epochs = total_epochs or (args.fl_rounds * args.local_epochs)
        self.dataset, self.train_loader = build_rtdetr_dataloader(
            data_yaml, "train", args, augment=True
        )
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.warmup_steps = round(args.warmup_epochs * self.steps_per_epoch)
        self.close_mosaic_epochs = int(args.close_mosaic_epochs)
        self._mosaic_closed = False
        self.global_step = 0
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.global_params = None
        self._snapshot_on_next_train = False
        self.train_losses = []
        self.train_components = []

    @property
    def amp_enabled(self):
        return bool(self.args.amp and self.device.type == "cuda")

    def reset_optimizer(self):
        self.optimizer = optim.AdamW(_optimizer_groups(self.rtdetr_lora, self.args))
        self.scheduler = WarmupCosine(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            min_lr_ratio=self.args.min_lr_ratio,
        )
        self.scaler = _make_grad_scaler(self.amp_enabled)

    def set_global_params(self):
        """Request a FedProx snapshot after this client is moved to its device."""
        self._snapshot_on_next_train = True

    def _capture_global_params(self):
        if self.args.fedprox_mu <= 0:
            self.global_params = None
            return
        payload_keys = set(self.rtdetr_lora.get_aggregation_state())
        self.global_params = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and name in payload_keys
        }

    def _proximal_term(self):
        if not self.global_params or self.args.fedprox_mu <= 0:
            return None
        squared_distance = None
        for name, parameter in self.model.named_parameters():
            if name not in self.global_params:
                continue
            term = (parameter - self.global_params[name]).pow(2).sum()
            squared_distance = term if squared_distance is None else squared_distance + term
        if squared_distance is None:
            return None
        return 0.5 * self.args.fedprox_mu * squared_distance

    def _prepare_training(self):
        self.model.to(self.device)
        if self.optimizer is None:
            self.reset_optimizer()
        if self._snapshot_on_next_train:
            self._capture_global_params()
            self._snapshot_on_next_train = False

    def _offload_if_possible(self):
        if self.client_id >= 0 and self.args.reset_optimizer_each_round:
            self.model.to("cpu")
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
            self.global_params = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _maybe_close_mosaic(self):
        """Mirror the official Trainer close-mosaic transition by effective epoch."""
        if (
            self._mosaic_closed
            or self.close_mosaic_epochs <= 0
            or self.global_epoch < self.total_epochs - self.close_mosaic_epochs
        ):
            return
        close_mosaic = getattr(self.dataset, "close_mosaic", None)
        hyp = getattr(self.dataset, "_study_hyp", None)
        if not callable(close_mosaic) or hyp is None:
            raise RuntimeError("Pinned RTDETRDataset no longer supports close_mosaic(hyp)")
        if hasattr(self.dataset, "mosaic"):
            self.dataset.mosaic = False
        hyp = copy(hyp)
        close_mosaic(hyp=hyp)
        self.dataset._study_hyp = hyp
        self._mosaic_closed = True
        print(
            f"  Client {self.client_id}: closed mosaic-family augmentations "
            f"before effective epoch {self.global_epoch + 1}"
        )

    def train_epoch(self, epochs: int = 1) -> dict:
        self._prepare_training()
        epoch_losses = []
        epoch_components = []
        epoch_lrs = []

        for _ in range(epochs):
            self._maybe_close_mosaic()
            self.model.train()
            self.rtdetr_lora.enforce_frozen_norm_eval()
            running_loss = 0.0
            batches = 0
            component_totals = defaultdict(float)

            for batch in self.train_loader:
                batch_data = _move_batch(batch, self.device)
                self.scheduler.set_step(self.global_step)
                self.optimizer.zero_grad(set_to_none=True)

                with _autocast(self.device, self.amp_enabled):
                    loss, components = _unpack_loss(self.model(batch_data))
                    proximal = self._proximal_term()
                    if proximal is not None:
                        loss = loss + proximal
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Client {self.client_id} encountered non-finite loss at "
                        f"global step {self.global_step}: {loss.detach().cpu().item()}"
                    )

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.rtdetr_lora.get_trainable_params(),
                        self.args.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.rtdetr_lora.get_trainable_params(),
                        self.args.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")

                running_loss += float(loss.detach().cpu())
                for key, value in components.items():
                    component_totals[key] += float(torch.as_tensor(value).detach().cpu())
                batches += 1
                self.global_step += 1

            if batches == 0:
                raise RuntimeError("Training epoch completed with zero batches")
            average_loss = running_loss / batches
            averages = {key: value / batches for key, value in component_totals.items()}
            epoch_losses.append(average_loss)
            epoch_components.append(averages)
            epoch_lrs.append([float(group["lr"]) for group in self.optimizer.param_groups])
            self.global_epoch += 1
            print(
                f"  Client {self.client_id} effective epoch {self.global_epoch}/{self.total_epochs}: "
                f"loss={average_loss:.5f}, lr_max={max(epoch_lrs[-1]):.3e}"
            )

        average = float(np.mean(epoch_losses))
        self.train_losses.extend(epoch_losses)
        self.train_components.extend(epoch_components)
        result = {
            "avg_loss": average,
            "epoch_losses": epoch_losses,
            "loss_components": epoch_components,
            "learning_rates": epoch_lrs,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "mosaic_closed": bool(self._mosaic_closed),
            "effective_close_mosaic_epochs": int(self.close_mosaic_epochs),
        }
        self._offload_if_possible()
        return result


def train_with_ultralytics(rtdetr_lora, data_yaml: str, epochs: int, args,
                           project_dir: str, run_name: str = "train",
                           validation_yamls: Optional[List[str]] = None) -> dict:
    """Manual RTDETRDataset loop used for both Full-FT and LoRA comparisons."""
    del run_name  # Kept in the public signature for compatibility.
    trainer = FLLocalTrainer(
        rtdetr_lora, data_yaml, args, client_id=-1, total_epochs=epochs
    )
    weights_dir = os.path.join(project_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_ap = -float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []
    selection_yamls = list(validation_yamls) if validation_yamls else [data_yaml]
    if not selection_yamls:
        raise ValueError("At least one validation dataset is required")

    for epoch in range(1, epochs + 1):
        train_result = trainer.train_epoch(epochs=1)
        row = {
            "epoch": epoch,
            "loss": train_result["epoch_losses"][0],
            "loss_components": train_result["loss_components"][0],
            "learning_rates": train_result["learning_rates"][0],
        }
        should_validate = epoch % max(1, args.val_interval) == 0 or epoch == epochs
        if should_validate:
            client_val_metrics = []
            for validation_index, validation_yaml in enumerate(selection_yamls):
                metrics = evaluate_model(rtdetr_lora, validation_yaml, args, split="val")
                client_val_metrics.append(metrics)
                label = "    Val" if len(selection_yamls) == 1 else f"    Client {validation_index} val"
                print_metrics(label, metrics)
            selection_metrics = {
                key: float(np.mean([metrics[key] for metrics in client_val_metrics]))
                for key in ("AP", "AP50", "AP75")
            }
            selection_metrics.update({
                "criterion": (
                    "single_client_validation"
                    if len(selection_yamls) == 1
                    else "macro_client_local_validation"
                ),
                "num_validation_partitions": len(selection_yamls),
                "per_partition": client_val_metrics,
            })
            row["val"] = selection_metrics
            if len(selection_yamls) > 1:
                print_metrics("    Macro client-local val", selection_metrics)
            if selection_metrics["AP"] > best_ap:
                best_ap = selection_metrics["AP"]
                best_epoch = epoch
                no_improvement = 0
                rtdetr_lora.save_model(
                    os.path.join(weights_dir, "best_full.pt"),
                    extra={"epoch": epoch, "val_metrics": selection_metrics},
                )
            else:
                no_improvement += 1

            if (
                args.visualize_interval > 0
                and epoch % args.visualize_interval == 0
            ):
                save_detection_samples(
                    rtdetr_lora,
                    data_yaml,
                    args,
                    os.path.join(project_dir, "detection_vis"),
                    tag=f"epoch_{epoch:03d}",
                    num_samples=args.vis_samples,
                    seed=args.seed,
                    splits=("val",),
                )
        history.append(row)
        if args.patience > 0 and no_improvement >= args.patience:
            print(f"[EarlyStop] No validation AP improvement for {args.patience} checks")
            break

    rtdetr_lora.save_model(
        os.path.join(weights_dir, "last_full.pt"),
        extra={"epoch": history[-1]["epoch"], "history": history},
    )
    best_path = os.path.join(weights_dir, "best_full.pt")
    if not os.path.isfile(best_path):
        raise RuntimeError("No best checkpoint was created")
    rtdetr_lora.load_model(best_path)
    return {
        "history": history,
        "best_val_ap": float(best_ap),
        "best_epoch": int(best_epoch),
        "epochs_executed": len(history),
        "best_checkpoint": best_path,
        "checkpoint_selection": (
            "single_client_validation_AP"
            if len(selection_yamls) == 1
            else "macro_client_local_validation_AP"
        ),
        "augmentation_protocol": rtdetr_augmentation_manifest(args),
        "effective_close_mosaic_epochs": int(trainer.close_mosaic_epochs),
    }


@contextlib.contextmanager
def _preserve_eval_state(rtdetr_lora, eval_device=None):
    """Keep Ultralytics standalone validation from poisoning trainable tensors.

    ``BaseValidator.__call__`` is wrapped in ``torch.inference_mode()`` in the
    pinned Ultralytics release.  Its PyTorch backend moves the supplied module
    to the requested device, casts it to FP32 and disables every parameter's
    gradient.  A device/dtype conversion performed there can replace parameter
    storage with inference tensors, which cannot later be made trainable.

    Perform the potentially allocating device/dtype conversion before entering
    Ultralytics, then restore device/dtype before restoring gradient flags.  The
    RT-DETR decoder's unregistered anchor cache is also restored because a
    validation-shape cache allocated in inference mode must not survive into a
    later training forward pass.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Evaluation state must be prepared outside torch.inference_mode()")

    ultralytics_wrapper = rtdetr_lora.ultralytics_model
    model = rtdetr_lora.model
    if ultralytics_wrapper.model is not model:
        raise RuntimeError("RTDETRLoRA and Ultralytics wrappers do not reference the same model")

    was_training = model.training
    first_parameter = next(model.parameters())
    original_device = first_parameter.device
    original_dtype = first_parameter.dtype
    target_device = torch.device(eval_device) if eval_device is not None else original_device
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
    try:
        # This must remain outside Ultralytics' inference-mode validator.  The
        # backend's subsequent .to(device)/.float() calls are then no-ops for
        # parameters rather than inference-tensor allocations.
        model.to(device=target_device, dtype=torch.float32)
        if original_fuse is not None:
            model.fuse = lambda *args, **kwargs: model
            fuse_patched = True
        yield original_device
    finally:
        # Model.val() is not expected to replace the module, but restore the
        # wrapper reference defensively before touching training state.
        ultralytics_wrapper.model = model
        if fuse_patched:
            model.fuse = original_fuse

        # Moving/casting first converts any accidentally allocated inference
        # storage back to ordinary tensors.  Only then is requires_grad=True
        # legal again.
        model.to(device=original_device, dtype=original_dtype)
        restored_parameters = dict(model.named_parameters())
        if tuple(restored_parameters) != tuple(original_parameters):
            raise RuntimeError("Ultralytics evaluation changed the model parameter keys")
        replaced = [
            name for name, parameter in restored_parameters.items()
            if parameter is not original_parameters[name]
        ]
        if replaced:
            raise RuntimeError(
                "Ultralytics evaluation replaced trainable Parameter objects: "
                f"{replaced[:5]}"
            )
        for name, parameter in restored_parameters.items():
            parameter.requires_grad_(grad_flags[name])

        for name, value in decoder_cache.items():
            if value is cache_sentinel:
                if hasattr(head, name):
                    delattr(head, name)
            else:
                setattr(head, name, value)
        model.train(was_training)
        rtdetr_lora.enforce_frozen_norm_eval()


def evaluate_model(rtdetr_lora, data_yaml: str, args, split: str = "val") -> dict:
    """Return Ultralytics COCO-style box AP metrics on a declared YAML split."""
    with _preserve_eval_state(rtdetr_lora, eval_device=args.device):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = rtdetr_lora.ultralytics_model.val(
                data=data_yaml,
                batch=args.batch_size,
                imgsz=args.img_size,
                device=args.device,
                workers=args.num_workers,
                split=split,
                verbose=False,
                plots=False,
                save_json=False,
                half=False,
                quantize=None,
            )
    if results is None or not hasattr(results, "box"):
        raise RuntimeError("Ultralytics validation returned no box metrics")

    box = results.box
    metrics = {
        "AP": float(box.map),
        "AP50": float(box.map50),
        "AP75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": {},
        "evaluator": "ultralytics_box_metrics_COCO_style_IoU_0.50_0.95",
    }
    names = getattr(rtdetr_lora.model, "names", {})
    class_indices = [int(value) for value in np.asarray(box.ap_class_index).reshape(-1)]
    per_ap = np.asarray(box.ap, dtype=np.float64).reshape(-1)
    per_ap50 = np.asarray(box.ap50, dtype=np.float64).reshape(-1)
    all_ap = np.asarray(getattr(box, "all_ap", []), dtype=np.float64)
    if len(per_ap) != len(class_indices) or len(per_ap50) != len(class_indices):
        raise RuntimeError(
            "Ultralytics per-class metric alignment changed: "
            f"class_indices={len(class_indices)}, AP={len(per_ap)}, AP50={len(per_ap50)}"
        )
    if all_ap.ndim != 2 or all_ap.shape[0] != len(class_indices) or all_ap.shape[1] < 6:
        raise RuntimeError(f"Unexpected Ultralytics all_ap shape: {all_ap.shape}")

    support = getattr(results, "nt_per_class", None)
    if support is None:
        support = getattr(box, "nt_per_class", None)
    if support is None:
        # Defensive pinned-API fallback. This is reached only if DetMetrics no
        # longer exposes target counts; it reads labels without augmentation.
        support = np.zeros(args.num_classes, dtype=np.int64)
        support_dataset = build_rtdetr_dataset(data_yaml, split, args, augment=False)
        for label in getattr(support_dataset, "labels", []):
            classes = np.asarray(label.get("cls", []), dtype=np.int64).reshape(-1)
            if classes.size:
                support += np.bincount(classes, minlength=args.num_classes)[: args.num_classes]
    support = np.asarray(support, dtype=np.int64).reshape(-1)
    if support.size != args.num_classes:
        raise RuntimeError(
            f"Ultralytics class-support length {support.size} != nc={args.num_classes}"
        )

    def class_name(class_index: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_index, class_index))
        if isinstance(names, (list, tuple)) and class_index < len(names):
            return str(names[class_index])
        return str(class_index)

    metrics["class_support"] = {
        class_name(class_index): int(support[class_index])
        for class_index in range(args.num_classes)
    }
    metrics["absent_classes"] = [
        class_name(class_index)
        for class_index in range(args.num_classes)
        if support[class_index] == 0
    ]
    for offset, class_index in enumerate(class_indices):
        metrics["per_class"][class_name(class_index)] = {
            "AP": float(per_ap[offset]),
            "AP50": float(per_ap50[offset]),
            "AP75": float(all_ap[offset, 5]),
            "support": int(support[class_index]),
        }
    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_ms_per_image"] = {
            str(key): float(value) for key, value in speed.items()
        }
    return metrics


def get_mia_losses(rtdetr_lora, data_yaml: str, args,
                   split: str = "train", max_samples: int = 500,
                   exclude_file_names=None) -> List[float]:
    """Extract image-level, ground-truth-matched RT-DETR losses (white-box MIA)."""
    dataset = build_rtdetr_dataset(data_yaml, split, args, augment=False)
    excluded = {
        str(value).replace("\\", "/").lstrip("/")
        for value in (exclude_file_names or ())
    }
    if excluded and split != "test":
        raise ValueError("MIA source-overlap exclusion is defined only for test non-members")
    marker = f"/{split}/images/"
    candidate_indices = []
    for index, image_path in enumerate(dataset.im_files):
        normalized = str(image_path).replace("\\", "/")
        relative_name = (
            normalized.rsplit(marker, 1)[1]
            if marker in normalized else os.path.basename(normalized)
        )
        if relative_name.lstrip("/") not in excluded:
            candidate_indices.append(index)
    sample_count = min(int(max_samples), len(candidate_indices))
    if sample_count < 2:
        raise ValueError(
            f"MIA split {split} has only {len(candidate_indices)} eligible images "
            f"after excluding {len(dataset) - len(candidate_indices)} source overlaps"
        )
    rng = np.random.default_rng(args.seed + (31 if split == "train" else 47))
    indices = sorted(
        int(value) for value in rng.choice(
            np.asarray(candidate_indices, dtype=np.int64),
            sample_count,
            replace=False,
        )
    )
    _, loader = build_rtdetr_dataloader(
        data_yaml,
        split,
        args,
        augment=False,
        batch_size=1,
        indices=indices,
    )

    device = torch.device(args.device)
    losses = []
    was_training = rtdetr_lora.model.training
    original_device = next(rtdetr_lora.model.parameters()).device
    rtdetr_lora.model.to(device)
    rtdetr_lora.model.eval()
    rtdetr_lora.enforce_frozen_norm_eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch_data = _move_batch(batch, device)
                loss, _ = _unpack_loss(rtdetr_lora.model(batch_data))
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite MIA loss on {split}: {value}")
                losses.append(value)
    finally:
        rtdetr_lora.model.train(was_training)
        rtdetr_lora.enforce_frozen_norm_eval()
        rtdetr_lora.model.to(original_device)
    if len(losses) != sample_count:
        raise RuntimeError(f"Expected {sample_count} MIA losses, collected {len(losses)}")
    return losses
