"""RT-DETR construction and exact LoRA/FedSA parameter routing."""

from __future__ import annotations

import copy
import hashlib
import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
import ultralytics
from ultralytics import RTDETR
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

from .lora import (
    LoRAConv2d,
    LoRALinear,
    LoRAMultiheadAttention,
    count_lora_params,
    get_lora_A_state_dict,
    get_lora_AB_state_dict,
    get_lora_B_state_dict,
    inject_lora_conv2d,
    inject_lora_linear,
    inject_lora_mha,
    set_lora_A_state_dict,
    set_lora_AB_state_dict,
    set_lora_B_state_dict,
)


PINNED_ULTRALYTICS_VERSION = "8.4.126"
TASK_HEAD_TOKENS = ("score_head", "class_embed")


def _cpu_clone_state(state: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _tensor_state_sha256(state: dict) -> str:
    """Deterministic fingerprint of tensor keys, dtypes, shapes and values."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_type_name(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_pretrained_rtdetr_source(source_model, model_file: str) -> dict:
    """Validate RT-DETR by architecture, not only its checkpoint container type.

    Ultralytics' official ``rtdetr-l.pt`` asset is serialized as the base
    ``DetectionModel`` even though its final module is ``RTDETRDecoder``.
    ``RTDETR(<pt>)`` restores that serialized class verbatim.  Requiring the
    newer ``RTDETRDetectionModel`` subclass therefore rejects a valid official
    checkpoint.  The decoder and YAML checks below accept that compatibility
    representation while still rejecting ordinary YOLO detection weights.
    """
    source_class = _qualified_type_name(source_model)
    if not isinstance(source_model, DetectionModel):
        raise TypeError(
            f"{model_file} loaded as {source_class}; expected an Ultralytics "
            "DetectionModel-compatible RT-DETR checkpoint"
        )

    layers = getattr(source_model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or len(layers) == 0:
        raise TypeError(
            f"{model_file} ({source_class}) has no non-empty module sequence"
        )
    source_head = layers[-1]
    head_class = _qualified_type_name(source_head)
    if not isinstance(source_head, RTDETRDecoder):
        raise TypeError(
            f"{model_file} loaded as {source_class} with final head {head_class}; "
            "expected RTDETRDecoder (ordinary YOLO checkpoints are not valid)"
        )

    source_yaml = getattr(source_model, "yaml", None)
    if not isinstance(source_yaml, dict):
        raise TypeError(f"{model_file} has no dictionary-valued RT-DETR YAML")
    for section in ("backbone", "head"):
        if not isinstance(source_yaml.get(section), list) or not source_yaml[section]:
            raise TypeError(
                f"{model_file} RT-DETR YAML has no non-empty {section!r} section"
            )
    try:
        yaml_nc = int(source_yaml["nc"])
        head_nc = int(source_head.nc)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise TypeError(
            f"{model_file} has no valid RT-DETR class-count metadata"
        ) from error
    if yaml_nc <= 0 or head_nc != yaml_nc:
        raise TypeError(
            f"{model_file} has inconsistent RT-DETR class counts: "
            f"yaml.nc={yaml_nc}, head.nc={head_nc}"
        )

    source_task = getattr(source_model, "task", None)
    if source_task not in (None, "detect"):
        raise TypeError(
            f"{model_file} declares task={source_task!r}, expected 'detect'"
        )
    return {
        "model_class": source_class,
        "head_class": head_class,
        "container_representation": (
            "rtdetr_detection_model"
            if isinstance(source_model, RTDETRDetectionModel)
            else "detection_model_with_rtdetr_decoder"
        ),
    }


class RTDETRLoRA:
    """A target-class RT-DETR model with explicit FL payload semantics.

    FedSA-LoRA payload = global LoRA A + global task/classification head.
    Every LoRA B remains in its client model and is never aggregated.

    Fixed Share-B payload = global LoRA B + global task/classification head.
    Every LoRA A remains in its client model and is never aggregated.  In both
    personalized methods A and B remain trainable; "fixed" describes the fixed
    communication role, not a frozen tensor.
    """

    def __init__(self, args, class_names: Optional[list] = None):
        if ultralytics.__version__ != PINNED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"This study pins ultralytics=={PINNED_ULTRALYTICS_VERSION}; "
                f"found {ultralytics.__version__}. Install requirements.txt first."
            )
        self.args = args
        self.ft_mode = args.fl_method
        self.num_classes = int(args.num_classes)
        self.class_names = class_names or [str(index) for index in range(self.num_classes)]
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length does not match num_classes")

        model_file = args.model_weights or f"{args.model_name}.pt"
        source_wrapper = RTDETR(model_file)
        source_model = source_wrapper.model
        source_descriptor = _validate_pretrained_rtdetr_source(
            source_model, model_file
        )
        self.pretrained_source_model_class = source_descriptor["model_class"]
        self.pretrained_source_head_class = source_descriptor["head_class"]
        self.pretrained_container_representation = source_descriptor[
            "container_representation"
        ]
        if not isinstance(source_model, RTDETRDetectionModel):
            print(
                "[Model] Accepted an RT-DETR checkpoint stored as DetectionModel "
                "because its final head is RTDETRDecoder; rebuilding the trainable "
                "model as RTDETRDetectionModel."
            )
        self.pretrained_state_sha256 = _tensor_state_sha256(source_model.state_dict())
        weight_candidates = (
            model_file,
            getattr(source_wrapper, "ckpt_path", None),
            getattr(source_model, "pt_path", None),
        )
        self.model_weight_path = next(
            (
                os.path.abspath(str(candidate))
                for candidate in weight_candidates
                if candidate and os.path.isfile(str(candidate))
            ),
            None,
        )
        self.model_weight_sha256 = (
            _file_sha256(self.model_weight_path) if self.model_weight_path else None
        )

        # Rebuild with nc=4 so every class-dependent component is correct,
        # including denoising_class_embed and RT-DETR-specific bias init.
        target_cfg = copy.deepcopy(source_model.yaml)
        target_cfg["nc"] = self.num_classes
        target_model = RTDETRDetectionModel(
            cfg=target_cfg,
            ch=3,
            nc=self.num_classes,
            verbose=False,
        )
        # DetectionTrainer normally attaches this outer attribute in
        # set_model_attributes().  This study uses a manual federated training
        # loop and calls RTDETRDetectionModel.loss() directly, whose pinned
        # init_criterion() reads self.nc.  Keep it synchronized explicitly.
        target_model.nc = self.num_classes
        target_model.names = {idx: name for idx, name in enumerate(self.class_names)}
        target_model.load(source_model, verbose=False)
        source_wrapper.model = target_model
        source_wrapper.overrides["model"] = model_file

        self.ultralytics_model = source_wrapper
        self.model = target_model
        self.set_class_names(self.class_names)
        self._validate_detection_head()
        self.initial_target_state_sha256 = _tensor_state_sha256(self.model.state_dict())

        self.lora_layers_mha = []
        self.lora_layers_linear = []
        self.lora_layers_conv = []
        if self.ft_mode in ("lora", "fedsa_lora", "fixed_share_b_lora"):
            self._inject_lora(args)
            self._freeze_base_params()
        elif self.ft_mode == "full_ft":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(f"Unknown fine-tuning mode: {self.ft_mode}")

        info = self.count_params()
        print(
            f"[Model] base={info['base_model_params']/1e6:.3f}M, "
            f"adapter={info['total_lora_params']/1e6:.3f}M, "
            f"trainable={info['trainable_params']/1e6:.3f}M, "
            f"one-way payload={info['communication_params']/1e6:.3f}M"
        )

    def _validate_detection_head(self):
        head = self.model.model[-1]
        if not isinstance(head, RTDETRDecoder):
            raise TypeError(
                f"Expected RTDETRDecoder head, found {_qualified_type_name(head)}"
            )
        try:
            yaml_nc = int(self.model.yaml["nc"])
            model_nc = int(self.model.nc)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "RT-DETR target model has no valid synchronized yaml.nc/model.nc"
            ) from error
        if (
            int(head.nc) != self.num_classes
            or yaml_nc != self.num_classes
            or model_nc != self.num_classes
        ):
            raise RuntimeError(
                "RT-DETR class rebuild failed: "
                f"yaml.nc={yaml_nc}, model.nc={model_nc}, head.nc={head.nc}"
            )
        if getattr(head, "denoising_class_embed", None) is not None:
            rows = int(head.denoising_class_embed.num_embeddings)
            if rows != self.num_classes:
                raise RuntimeError(
                    f"denoising_class_embed has {rows} rows, expected {self.num_classes}"
                )

    def set_class_names(self, class_names: list):
        if len(class_names) != self.num_classes:
            raise ValueError("class_names length does not match num_classes")
        self.class_names = list(class_names)
        names = {idx: name for idx, name in enumerate(class_names)}
        self.model.names = names
        self.ultralytics_model.model.names = names

    @staticmethod
    def _is_task_head_key(name: str) -> bool:
        return any(token in name for token in TASK_HEAD_TOKENS)

    def _decoder_targets(self):
        mha_targets, linear_targets = [], []
        for name, module in self.model.named_modules():
            if ".decoder.layers." not in name:
                continue
            if name.endswith(".self_attn") and isinstance(module, nn.MultiheadAttention):
                mha_targets.append(name)
            if name.endswith(".cross_attn.value_proj") and isinstance(module, nn.Linear):
                linear_targets.append(name)
        return sorted(mha_targets), sorted(linear_targets)

    def _backbone_conv_targets(self):
        backbone_cfg = self.model.yaml.get("backbone")
        if not isinstance(backbone_cfg, list):
            raise RuntimeError("RT-DETR yaml has no list-valued backbone definition")
        prefixes = tuple(f"model.{index}" for index in range(len(backbone_cfg)))
        targets = []
        for name, module in self.model.named_modules():
            in_backbone = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
            if not in_backbone or not isinstance(module, nn.Conv2d):
                continue
            if module.groups != 1:
                continue
            if (
                module.in_channels >= self.args.backbone_min_channels
                and module.out_channels >= self.args.backbone_min_channels
            ):
                targets.append(name)
        return sorted(targets)

    def _inject_lora(self, args):
        if args.apply_lora_decoder:
            mha_targets, linear_targets = self._decoder_targets()
            if not mha_targets:
                raise RuntimeError("No fused self-attention Q/K/V targets found in RT-DETR decoder")
            if not linear_targets:
                raise RuntimeError("No deformable cross-attention value projection targets found")
            self.lora_layers_mha = inject_lora_mha(
                self.model, mha_targets, args.lora_rank, args.lora_alpha, args.lora_dropout
            )
            self.lora_layers_linear = inject_lora_linear(
                self.model, linear_targets, args.lora_rank, args.lora_alpha, args.lora_dropout
            )

        if args.apply_lora_backbone:
            conv_targets = self._backbone_conv_targets()
            if not conv_targets:
                raise RuntimeError("No eligible CNN backbone convolutions found for LoRA")
            self.lora_layers_conv = inject_lora_conv2d(
                self.model,
                conv_targets,
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
                args.backbone_min_channels,
            )
            if not self.lora_layers_conv:
                raise RuntimeError("CNN backbone LoRA target filtering removed every target")

        print(
            f"[LoRA] QKV-MHA={len(self.lora_layers_mha)}, "
            f"cross-value={len(self.lora_layers_linear)}, "
            f"backbone-conv={len(self.lora_layers_conv)}"
        )

    def _freeze_base_params(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.model.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                parameter.requires_grad_(True)
            elif self._is_task_head_key(name):
                parameter.requires_grad_(True)
        self.enforce_frozen_norm_eval()

    def enforce_frozen_norm_eval(self):
        """Freeze running statistics without turning off their buffers."""
        if self.ft_mode == "full_ft":
            return
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()

    def get_trainable_params(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _get_task_head_state(self):
        return _cpu_clone_state({
            key: value for key, value in self.model.state_dict().items()
            if self._is_task_head_key(key)
        })

    def _set_task_head_state(self, state: dict):
        current = self.model.state_dict()
        expected = {key for key in current if self._is_task_head_key(key)}
        supplied = set(state)
        if expected != supplied:
            raise RuntimeError(
                f"Task-head state mismatch; missing={sorted(expected - supplied)}, "
                f"unexpected={sorted(supplied - expected)}"
            )
        with torch.no_grad():
            for key, value in state.items():
                target = current[key]
                if tuple(target.shape) != tuple(value.shape):
                    raise RuntimeError(
                        f"Task-head shape mismatch for {key}: "
                        f"{tuple(target.shape)} != {tuple(value.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def shared_lora_factor_role(self) -> Optional[str]:
        """Return the LoRA factor(s) carried by the federated payload."""
        if self.ft_mode == "full_ft":
            return None
        if self.ft_mode == "lora":
            return "A+B"
        if self.ft_mode == "fedsa_lora":
            return "A"
        if self.ft_mode == "fixed_share_b_lora":
            return "B"
        raise ValueError(self.ft_mode)

    def local_personalized_factor_role(self) -> Optional[str]:
        """Return the trainable LoRA factor retained separately by each client."""
        if self.ft_mode in ("full_ft", "lora"):
            return None
        if self.ft_mode == "fedsa_lora":
            return "B"
        if self.ft_mode == "fixed_share_b_lora":
            return "A"
        raise ValueError(self.ft_mode)

    def federated_payload_policy(self) -> str:
        policies = {
            "full_ft": "global_full_model_state",
            "lora": "global_A_B_plus_global_task_head",
            "fedsa_lora": "global_A_plus_global_task_head__local_B",
            "fixed_share_b_lora": "global_B_plus_global_task_head__local_A",
        }
        try:
            return policies[self.ft_mode]
        except KeyError as error:
            raise ValueError(self.ft_mode) from error

    def get_aggregation_state(self) -> dict:
        if self.ft_mode == "full_ft":
            return _cpu_clone_state(self.model.state_dict())
        if self.ft_mode == "lora":
            state = get_lora_AB_state_dict(self.model)
        elif self.ft_mode == "fedsa_lora":
            state = get_lora_A_state_dict(self.model)
        elif self.ft_mode == "fixed_share_b_lora":
            state = get_lora_B_state_dict(self.model)
        else:
            raise ValueError(self.ft_mode)
        state.update(self._get_task_head_state())
        return state

    def set_aggregation_state(self, state: dict):
        if self.ft_mode == "full_ft":
            incompatible = self.model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Full state mismatch: {incompatible}")
            return
        head_state = {key: value for key, value in state.items() if self._is_task_head_key(key)}
        lora_state = {key: value for key, value in state.items() if key not in head_state}
        if self.ft_mode == "lora":
            set_lora_AB_state_dict(self.model, lora_state)
        elif self.ft_mode == "fedsa_lora":
            set_lora_A_state_dict(self.model, lora_state)
        elif self.ft_mode == "fixed_share_b_lora":
            set_lora_B_state_dict(self.model, lora_state)
        else:
            raise ValueError(self.ft_mode)
        self._set_task_head_state(head_state)

    def get_local_personalized_state(self) -> dict:
        """Return the client-retained factor, or an empty state for shared methods."""
        role = self.local_personalized_factor_role()
        if role == "A":
            return get_lora_A_state_dict(self.model)
        if role == "B":
            return get_lora_B_state_dict(self.model)
        return {}

    def set_local_personalized_state(self, state: dict):
        """Restore the client-retained factor with strict key and shape checks."""
        role = self.local_personalized_factor_role()
        if role is None:
            if state:
                raise RuntimeError(
                    f"{self.ft_mode} has no client-local personalized LoRA state"
                )
            return
        if role == "A":
            set_lora_A_state_dict(self.model, state)
        else:
            set_lora_B_state_dict(self.model, state)

    def get_local_B_state(self):
        """Backward-compatible FedSA-specific accessor."""
        if self.local_personalized_factor_role() != "B":
            raise RuntimeError(
                f"{self.ft_mode} does not retain LoRA B as its client-local factor"
            )
        return get_lora_B_state_dict(self.model)

    def set_local_B_state(self, state: dict):
        """Backward-compatible FedSA-specific setter."""
        if self.local_personalized_factor_role() != "B":
            raise RuntimeError(
                f"{self.ft_mode} does not retain LoRA B as its client-local factor"
            )
        set_lora_B_state_dict(self.model, state)

    def get_local_A_state(self):
        """Backward-compatible explicit accessor for Fixed Share-B checkpoints."""
        if self.local_personalized_factor_role() != "A":
            raise RuntimeError(
                f"{self.ft_mode} does not retain LoRA A as its client-local factor"
            )
        return get_lora_A_state_dict(self.model)

    def set_local_A_state(self, state: dict):
        """Backward-compatible explicit setter for Fixed Share-B checkpoints."""
        if self.local_personalized_factor_role() != "A":
            raise RuntimeError(
                f"{self.ft_mode} does not retain LoRA A as its client-local factor"
            )
        set_lora_A_state_dict(self.model, state)

    def count_params(self) -> dict:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        lora = count_lora_params(self.model)
        base_parameters = total - lora["total_lora_params"]
        task_head_parameters = sum(
            parameter.numel() for name, parameter in self.model.named_parameters()
            if self._is_task_head_key(name)
        )

        aggregation_state = self.get_aggregation_state()
        communication_params = sum(tensor.numel() for tensor in aggregation_state.values())
        communication_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in aggregation_state.values()
        )
        full_reference_state = {
            key: value for key, value in self.model.state_dict().items()
            if not key.endswith(".lora_A") and not key.endswith(".lora_B")
        }
        full_payload_params = sum(tensor.numel() for tensor in full_reference_state.values())
        trainable_ratio = 100.0 * trainable / base_parameters if base_parameters else 0.0
        parameter_saving = 100.0 * (1.0 - trainable / base_parameters) if base_parameters else 0.0
        communication_ratio = (
            100.0 * communication_params / full_payload_params if full_payload_params else 0.0
        )
        communication_saving = 100.0 - communication_ratio
        return {
            "total_params": int(total),
            "base_model_params": int(base_parameters),
            "trainable_params": int(trainable),
            "frozen_params": int(total - trainable),
            "task_head_params": int(task_head_parameters),
            **lora,
            "communication_params": int(communication_params),
            "communication_bytes": int(communication_bytes),
            "full_ft_payload_params": int(full_payload_params),
            "trainable_ratio_pct": float(trainable_ratio),
            "parameter_saving_pct": float(parameter_saving),
            "communication_ratio_pct": float(communication_ratio),
            "communication_saving_pct": float(communication_saving),
            "communication_efficiency_definition": (
                "tensor_element_saving_vs_full_ft_payload_pct"
            ),
            # Backward-compatible aliases now mean savings, not fraction used.
            "param_efficiency": float(parameter_saving),
        }

    def architecture_manifest(self) -> dict:
        adapters = []
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d, LoRAMultiheadAttention)):
                adapters.append({
                    "name": name,
                    "type": type(module).__name__,
                    "lora_A_shape": list(module.lora_A.shape),
                    "lora_B_shape": list(module.lora_B.shape),
                    "rank": module.rank,
                    "scaling": module.scaling,
                })
        return {
            "ultralytics_version": ultralytics.__version__,
            "model_name": self.args.model_name,
            "model_weight_path": self.model_weight_path,
            "model_weight_sha256": self.model_weight_sha256,
            "pretrained_source_model_class": self.pretrained_source_model_class,
            "pretrained_source_head_class": self.pretrained_source_head_class,
            "pretrained_container_representation": (
                self.pretrained_container_representation
            ),
            "pretrained_tensor_state_sha256": self.pretrained_state_sha256,
            "initial_target_tensor_state_sha256": self.initial_target_state_sha256,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "fine_tuning_mode": self.ft_mode,
            "federated_payload_policy": self.federated_payload_policy(),
            "shared_lora_factor_role": self.shared_lora_factor_role(),
            "client_local_lora_factor_role": self.local_personalized_factor_role(),
            # Retain the old field for schema consumers, but do not falsely label
            # full FT, ordinary LoRA, or Fixed Share-B as FedSA.
            "fedsa_payload_policy": (
                self.federated_payload_policy()
                if self.ft_mode == "fedsa_lora" else None
            ),
            "adapters": adapters,
            "parameter_counts": self.count_params(),
        }

    def save_model(self, path: str, extra: Optional[dict] = None):
        payload = {
            "schema_version": 2,
            "ft_mode": self.ft_mode,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "ultralytics_version": ultralytics.__version__,
            "model_state": _cpu_clone_state(self.model.state_dict()),
            "extra": extra or {},
        }
        temporary = path + ".tmp"
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def load_model(self, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        checks = {
            "ft_mode": (payload["ft_mode"], self.ft_mode),
            "num_classes": (int(payload["num_classes"]), self.num_classes),
            "class_names": (list(payload["class_names"]), self.class_names),
            "ultralytics_version": (
                payload["ultralytics_version"], ultralytics.__version__
            ),
        }
        mismatches = [
            f"{key}: checkpoint={left!r}, current={right!r}"
            for key, (left, right) in checks.items() if left != right
        ]
        if mismatches:
            raise ValueError("Checkpoint mismatch: " + "; ".join(mismatches))
        incompatible = self.model.load_state_dict(payload["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint state mismatch: {incompatible}")
        return payload.get("extra", {})
