"""Strict FedAvg orchestration, personalized checkpoints and FL evaluation."""

from __future__ import annotations

import copy
import hashlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from models.rtdetr_lora import RTDETRLoRA
from trainers.trainer import (
    FLLocalTrainer,
    evaluate_model,
    get_mia_losses,
    print_metrics,
    rtdetr_augmentation_manifest,
)
from utils.mia import client_attack_seed, compute_mia_metrics
from utils.visualization import (
    plot_client_data_distribution,
    plot_fl_metrics,
    plot_training_loss,
    save_detection_samples,
    save_results_json,
)


FEDERATED_CHECKPOINT_SCHEMA = 5
SUPPORTED_FEDERATED_CHECKPOINT_SCHEMAS = (4, FEDERATED_CHECKPOINT_SCHEMA)
FEDERATED_RESULT_SCHEMA = 2
REPORT_METRICS = ("AP", "AP50", "AP75")
NONFLOATING_STATE_POLICY = "retain_previous_server_value"


def _cpu_clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state:
        raise ValueError("A federated state must be a non-empty tensor dictionary")
    cloned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Invalid state entry {key!r}: expected str -> Tensor")
        cloned[key] = value.detach().cpu().clone()
    return cloned


def _payload_statistics(state: Dict[str, torch.Tensor]) -> dict:
    """Count the exact serialized tensor payload, excluding container overhead."""
    if not isinstance(state, dict) or not state:
        raise ValueError("A communication payload must be a non-empty tensor dictionary")
    dtype_breakdown = {}
    parameters = 0
    byte_count = 0
    nonfloating_keys = []
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid payload entry {key!r}: expected str -> Tensor")
        tensor_params = int(tensor.numel())
        tensor_bytes = int(tensor_params * tensor.element_size())
        parameters += tensor_params
        byte_count += tensor_bytes
        dtype_key = str(tensor.dtype)
        bucket = dtype_breakdown.setdefault(dtype_key, {"params": 0, "bytes": 0})
        bucket["params"] += tensor_params
        bucket["bytes"] += tensor_bytes
        if not (tensor.is_floating_point() or tensor.is_complex()):
            nonfloating_keys.append(key)
    return {
        "params": int(parameters),
        "bytes": int(byte_count),
        "mb": float(byte_count / 1_000_000.0),
        "tensor_count": int(len(state)),
        "dtype_breakdown": dtype_breakdown,
        "nonfloating_tensor_count": int(len(nonfloating_keys)),
        "nonfloating_keys": sorted(nonfloating_keys),
    }


def _scale_payload(payload: dict, multiplier: int) -> dict:
    multiplier = int(multiplier)
    return {
        "params": int(payload["params"] * multiplier),
        "bytes": int(payload["bytes"] * multiplier),
        "mb": float(payload["bytes"] * multiplier / 1_000_000.0),
        "transmissions": multiplier,
    }


def _communication_accounting(
    state: Dict[str, torch.Tensor], num_clients: int, rounds_executed: int
) -> dict:
    one_way = _payload_statistics(state)
    round_upload = _scale_payload(one_way, num_clients)
    round_download = _scale_payload(one_way, num_clients)
    round_total = _scale_payload(one_way, 2 * num_clients)
    cumulative_upload = _scale_payload(one_way, num_clients * rounds_executed)
    cumulative_download = _scale_payload(one_way, num_clients * rounds_executed)
    cumulative_total = _scale_payload(one_way, 2 * num_clients * rounds_executed)
    result = {
        "unit": "decimal_MB_1e6_bytes",
        "scope": "model_tensor_payload_only",
        "initial_pretrained_distribution_included": False,
        "initial_pretrained_or_base_provisioning_included": False,
        "final_post_aggregation_download_included": bool(rounds_executed > 0),
        "validation_selected_checkpoint_redistribution_included": False,
        "round_definition": (
            "num_clients uploads plus num_clients post-aggregation downloads; "
            "the final round download is included"
        ),
        "checkpoint_and_protocol_overhead_included": False,
        "num_clients": int(num_clients),
        "rounds_executed": int(rounds_executed),
        "one_client_one_way": one_way,
        "round_upload": round_upload,
        "round_download": round_download,
        "round_total": round_total,
        "cumulative_upload": cumulative_upload,
        "cumulative_download": cumulative_download,
        "cumulative_total": cumulative_total,
    }
    # Scalar aliases keep result aggregation scripts simple while the structured
    # records above retain exact byte/parameter/transmission counts.
    result.update({
        "one_client_one_way_mb": one_way["mb"],
        "one_way_client_mb": one_way["mb"],
        "round_upload_mb": round_upload["mb"],
        "round_download_mb": round_download["mb"],
        "round_total_mb": round_total["mb"],
        "system_round_total_mb": round_total["mb"],
        "cumulative_upload_mb": cumulative_upload["mb"],
        "cumulative_download_mb": cumulative_download["mb"],
        "cumulative_total_mb": cumulative_total["mb"],
        "total_mb": cumulative_total["mb"],
    })
    return result


def _metric_summary(
    metrics: Sequence[dict], sample_counts: Optional[Sequence[int]] = None
) -> dict:
    if len(metrics) < 2:
        raise ValueError("Client disparity requires at least two client metric records")
    if sample_counts is None:
        weights = np.full(len(metrics), 1.0 / len(metrics), dtype=np.float64)
        weighting = "uniform"
    else:
        counts = np.asarray(sample_counts, dtype=np.float64)
        if (
            counts.shape != (len(metrics),)
            or not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or counts.sum() <= 0
        ):
            raise ValueError(f"Invalid client evaluation sample counts: {counts.tolist()}")
        weights = counts / counts.sum()
        weighting = "client_evaluation_image_count"
    summary = {
        "num_clients": int(len(metrics)),
        "std_definition": "sample_ddof_1",
        "weighted_mean_definition": weighting,
        "definition": (
            "macro_mean is the unweighted client mean; client_sample_sd is the "
            "across-client sample standard deviation (ddof=1); weighted_mean uses "
            "client evaluation image counts when supplied and is not pooled COCO AP."
        ),
    }
    for key in REPORT_METRICS:
        values = np.asarray([float(record[key]) for record in metrics], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Non-finite client metric {key}: {values.tolist()}")
        worst_index = int(np.argmin(values))
        best_index = int(np.argmax(values))
        summary[key] = {
            "macro_mean": float(values.mean()),
            "weighted_mean": float(np.dot(weights, values)),
            "client_sample_sd": float(values.std(ddof=1)),
            "sample_std": float(values.std(ddof=1)),
            "worst_client": float(values[worst_index]),
            "worst": float(values[worst_index]),
            "worst_client_id": worst_index,
            "best_client": float(values[best_index]),
            "best": float(values[best_index]),
            "best_client_id": best_index,
            "max_min_gap": float(values[best_index] - values[worst_index]),
            "gap": float(values[best_index] - values[worst_index]),
        }
    return summary


def _mia_summary(per_client: Sequence[dict]) -> dict:
    if len(per_client) < 2:
        raise ValueError("Federated MIA summary requires at least two clients")
    summary = {"num_clients": int(len(per_client)), "std_definition": "sample_ddof_1"}
    for key in ("auc_roc", "tpr_at_1fpr", "asr"):
        values = np.asarray([float(item["metrics"][key]) for item in per_client])
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Non-finite MIA metric {key}: {values.tolist()}")
        summary[key] = {
            "macro_mean": float(values.mean()),
            "client_sample_sd": float(values.std(ddof=1)),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


class FLServer:
    """All-client, sample-weighted FedAvg with strict state validation.

    Floating and complex tensors are averaged with weights ``n_i / sum(n_i)``.
    Integer and boolean buffers (for example ``num_batches_tracked``) are not
    mathematically averageable, so the previous server value is retained and
    broadcast. The policy is explicit in every result/checkpoint.
    """

    def __init__(self, fl_method: str, num_clients: int):
        if fl_method not in (
            "full_ft", "lora", "fedsa_lora", "fixed_share_b_lora"
        ):
            raise ValueError(f"Unsupported federated method: {fl_method!r}")
        if num_clients < 2:
            raise ValueError("Federated learning requires at least two clients")
        self.fl_method = str(fl_method)
        self.num_clients = int(num_clients)
        self.global_state: Optional[Dict[str, torch.Tensor]] = None
        self.last_normalized_weights: Optional[List[float]] = None

    def initialize_global_state(self, initial_state: Dict[str, torch.Tensor]):
        self.global_state = _cpu_clone_state(initial_state)

    def _validate_inputs(
        self,
        client_states: Sequence[Dict[str, torch.Tensor]],
        client_sample_counts: Sequence[int],
    ) -> np.ndarray:
        if self.global_state is None:
            raise RuntimeError("initialize_global_state() must be called before aggregate()")
        if len(client_states) != self.num_clients:
            raise ValueError(
                f"Expected {self.num_clients} client states, received {len(client_states)}"
            )
        if len(client_sample_counts) != self.num_clients:
            raise ValueError(
                f"Expected {self.num_clients} sample counts, received {len(client_sample_counts)}"
            )
        counts = np.asarray(client_sample_counts, dtype=np.float64)
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError(f"Client sample counts must be finite and positive: {counts.tolist()}")
        if not np.all(counts == np.floor(counts)):
            raise ValueError(f"Client sample counts must be integers: {counts.tolist()}")

        expected_keys = set(self.global_state)
        for client_id, state in enumerate(client_states):
            if not isinstance(state, dict):
                raise TypeError(f"Client {client_id} state is not a dictionary")
            supplied_keys = set(state)
            if supplied_keys != expected_keys:
                raise RuntimeError(
                    f"Client {client_id} state keys mismatch; "
                    f"missing={sorted(expected_keys - supplied_keys)}, "
                    f"unexpected={sorted(supplied_keys - expected_keys)}"
                )
            for key in sorted(expected_keys):
                value = state[key]
                reference = self.global_state[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Client {client_id} state {key!r} is not a Tensor")
                if value.shape != reference.shape or value.dtype != reference.dtype:
                    raise RuntimeError(
                        f"Client {client_id} state mismatch for {key}: "
                        f"shape/dtype={tuple(value.shape)}/{value.dtype}, "
                        f"expected={tuple(reference.shape)}/{reference.dtype}"
                    )
                if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                    raise FloatingPointError(f"Client {client_id} supplied non-finite tensor {key}")
        return counts / counts.sum()

    def aggregate(
        self,
        client_states: Sequence[Dict[str, torch.Tensor]],
        client_sample_counts: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        normalized = self._validate_inputs(client_states, client_sample_counts)
        self.last_normalized_weights = [float(value) for value in normalized]
        aggregated = {}
        for key in sorted(self.global_state):
            reference = self.global_state[key]
            if not (reference.is_floating_point() or reference.is_complex()):
                aggregated[key] = reference.detach().cpu().clone()
                continue

            if reference.is_complex():
                accumulation_dtype = (
                    torch.complex128 if reference.dtype == torch.complex128 else torch.complex64
                )
            else:
                accumulation_dtype = (
                    torch.float64 if reference.dtype == torch.float64 else torch.float32
                )
            accumulator = torch.zeros(reference.shape, dtype=accumulation_dtype, device="cpu")
            for weight, state in zip(normalized, client_states):
                accumulator.add_(
                    state[key].detach().to(device="cpu", dtype=accumulation_dtype),
                    alpha=float(weight),
                )
            if not torch.isfinite(accumulator).all():
                raise FloatingPointError(f"FedAvg produced a non-finite tensor for {key}")
            aggregated[key] = accumulator.to(dtype=reference.dtype)

        self.global_state = aggregated
        return self.get_global_state()

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        if self.global_state is None:
            raise RuntimeError("Global state is not initialized")
        return _cpu_clone_state(self.global_state)

    def compute_communication_cost(self, state: Dict[str, torch.Tensor]) -> dict:
        """Backward-compatible one-client, one-way payload description."""
        payload = _payload_statistics(state)
        return {**payload, "size_mb": payload["mb"]}


def _validate_data_info(args, data_info: dict):
    required = (
        "client_yamls",
        "client_sizes",
        "client_eval_sizes",
        "full_yaml",
        "class_names",
        "client_splits",
        "train_coco",
        "split_manifest_sha256",
    )
    missing = [key for key in required if key not in data_info]
    if missing:
        raise KeyError(f"data_info is missing required fields: {missing}")
    for key in ("client_yamls", "client_sizes", "client_eval_sizes", "client_splits"):
        if len(data_info[key]) != args.num_clients:
            raise ValueError(
                f"data_info[{key!r}] has {len(data_info[key])} clients, "
                f"expected {args.num_clients}"
            )
    if len(data_info["class_names"]) != args.num_classes:
        raise ValueError("data_info class count does not match --num_classes")
    for client_id, (train_size, train_ids, evaluation_sizes) in enumerate(zip(
        data_info["client_sizes"], data_info["client_splits"], data_info["client_eval_sizes"]
    )):
        if int(train_size) <= 0 or int(train_size) != len(train_ids):
            raise ValueError(
                f"Client {client_id} train size/manifest mismatch: "
                f"size={train_size}, ids={len(train_ids)}"
            )
        for split in ("val", "test"):
            if int(evaluation_sizes.get(split, 0)) <= 0:
                raise ValueError(f"Client {client_id} has no {split} evaluation images")
    split_digest = str(data_info["split_manifest_sha256"]).lower()
    if len(split_digest) != 64 or any(character not in "0123456789abcdef" for character in split_digest):
        raise ValueError("data_info split_manifest_sha256 is not a valid SHA-256 digest")


def _frozen_base_sha256(
    model: RTDETRLoRA, *, refresh: bool = False
) -> Optional[str]:
    """Fingerprint LoRA-untransmitted weights required to reconstruct a checkpoint."""
    if model.ft_mode == "full_ft":
        return None
    cached = getattr(model, "_federated_frozen_base_sha256", None)
    if cached is not None and not refresh:
        return str(cached)

    # Together the federated payload and optional client-local factor must cover
    # every trainable adapter/task tensor.  Using the generic personalized state
    # is essential for Fixed Share-B, whose local factor is A rather than B.
    covered_keys = (
        set(model.get_aggregation_state())
        | set(model.get_local_personalized_state())
    )
    full_state = model.model.state_dict()
    frozen_keys = sorted(set(full_state) - covered_keys)
    if not frozen_keys:
        raise RuntimeError("No frozen base tensors remain outside the LoRA checkpoint payload")

    digest = hashlib.sha256()
    for key in frozen_keys:
        tensor = full_state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    value = digest.hexdigest()
    setattr(model, "_federated_frozen_base_sha256", value)
    return value


def _new_client_models(args, data_info: dict) -> List[RTDETRLoRA]:
    base_model = RTDETRLoRA(args, class_names=data_info["class_names"])
    _frozen_base_sha256(base_model)
    models = [base_model]
    models.extend(copy.deepcopy(base_model) for _ in range(1, args.num_clients))
    for client_id, model in enumerate(models):
        if model.ultralytics_model.model is not model.model:
            raise RuntimeError(
                f"Client {client_id} deepcopy broke the RTDETR wrapper/model reference"
            )
    return models


def _atomic_torch_save(payload: dict, path: str):
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _experiment_manifest(args) -> dict:
    keys = (
        "fl_method", "model_name", "model_weights", "num_clients", "fl_rounds",
        "local_epochs", "partition", "dirichlet_alpha", "lora_rank", "lora_alpha",
        "lora_dropout", "batch_size", "img_size", "lr", "head_lr", "weight_decay",
        "backbone_lr_ratio", "warmup_epochs", "min_lr_ratio", "grad_clip_norm",
        "close_mosaic_epochs",
        "fedprox_mu", "reset_optimizer_each_round", "amp", "patience", "val_interval",
        "seed", "partition_seed", "num_workers", "cross_client_eval",
        "visualize_interval", "vis_samples", "mia_max_samples",
        "mia_calibration_fraction", "apply_lora_backbone", "apply_lora_decoder",
        "backbone_min_channels",
    )
    manifest = {key: getattr(args, key, None) for key in keys}
    if args.partition != "dirichlet":
        # ``dirichlet_alpha`` is an argparse default, not a property of the
        # explicit label-blind IID partition.  Keeping 0.4 here would falsely
        # describe the IID experiment as having two partition protocols.
        manifest["dirichlet_alpha"] = None
    manifest.update({
        "optimizer": "AdamW",
        "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
        "client_participation": "all_clients_every_round",
        "aggregation_weighting": "local_train_image_count",
        "nonfloating_state_policy": NONFLOATING_STATE_POLICY,
        "validation_frequency_rounds": 1,
        "selection_criterion": "macro_client_local_validation_AP",
        "local_epoch_budget_per_client": int(args.fl_rounds * args.local_epochs),
        "communication_round_convention": (
            "num_clients_uploads_plus_num_clients_post_aggregation_downloads"
        ),
        "augmentation_protocol": rtdetr_augmentation_manifest(args),
        "effective_close_mosaic_epochs": int(args.close_mosaic_epochs),
        "mia_attack": "image_level_ground_truth_matched_detection_loss_threshold",
        "mia_nonmember_source_policy": (
            "exclude_test_source_components_present_in_any_train_client"
        ),
        "mia_bootstrap": {
            "method": "stratified_nonparametric_full_attack_pipeline_percentile",
            "resamples": 1000,
            "confidence_level": 0.95,
            "recalibrates_direction_and_threshold": True,
        },
    })
    return manifest


def _checkpoint_compatibility_manifest(args) -> dict:
    """Configuration fields that must match before checkpoint tensors are loaded."""
    uses_lora = args.fl_method != "full_ft"
    return {
        "fl_method": args.fl_method,
        "model_name": args.model_name,
        "num_classes": int(args.num_classes),
        "num_clients": int(args.num_clients),
        "fl_rounds": int(args.fl_rounds),
        "local_epochs": int(args.local_epochs),
        "batch_size": int(args.batch_size),
        "img_size": int(args.img_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "head_lr": float(args.head_lr),
        "backbone_lr_ratio": float(args.backbone_lr_ratio),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": float(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
        "grad_clip_norm": float(args.grad_clip_norm),
        "close_mosaic_epochs": int(args.close_mosaic_epochs),
        "fedprox_mu": float(args.fedprox_mu),
        "reset_optimizer_each_round": bool(args.reset_optimizer_each_round),
        "amp": bool(args.amp),
        "augmentation_protocol": rtdetr_augmentation_manifest(args),
        "seed": int(args.seed),
        "partition_seed": int(args.partition_seed),
        "partition": args.partition,
        "dirichlet_alpha": (
            float(args.dirichlet_alpha) if args.partition == "dirichlet" else None
        ),
        "lora_rank": int(args.lora_rank) if uses_lora else None,
        "lora_alpha": float(args.lora_alpha) if uses_lora else None,
        "lora_dropout": float(args.lora_dropout) if uses_lora else None,
        "apply_lora_backbone": bool(args.apply_lora_backbone) if uses_lora else None,
        "apply_lora_decoder": bool(args.apply_lora_decoder) if uses_lora else None,
        "backbone_min_channels": int(args.backbone_min_channels) if uses_lora else None,
    }


def _make_checkpoint(
    *, args, data_info: dict, client_models: Sequence[RTDETRLoRA], server: FLServer,
    client_trainers: Sequence[FLLocalTrainer], round_number: int, selection: str,
    local_val: Sequence[dict], local_val_summary: dict, round_metrics: Sequence[dict],
    training_rounds_executed: int,
) -> dict:
    local_factor_role = client_models[0].local_personalized_factor_role()
    local_personalized_states = None
    if local_factor_role is not None:
        observed_roles = {
            model.local_personalized_factor_role() for model in client_models
        }
        if observed_roles != {local_factor_role}:
            raise RuntimeError(
                "Client models disagree about the local LoRA factor role: "
                f"{sorted(str(value) for value in observed_roles)}"
            )
        local_personalized_states = [
            model.get_local_personalized_state() for model in client_models
        ]
    return {
        "schema_version": FEDERATED_CHECKPOINT_SCHEMA,
        "checkpoint_kind": "federated_personalized",
        "resume_capability": "evaluation_only_no_optimizer_scheduler_or_rng_state",
        "selection": selection,
        "round": int(round_number),
        "training_rounds_executed": int(training_rounds_executed),
        "fl_method": args.fl_method,
        "num_clients": int(args.num_clients),
        "class_names": list(data_info["class_names"]),
        "split_manifest_sha256": str(data_info["split_manifest_sha256"]).lower(),
        "client_sample_counts": [int(value) for value in data_info["client_sizes"]],
        "normalized_fedavg_weights": list(server.last_normalized_weights or []),
        "nonfloating_state_policy": NONFLOATING_STATE_POLICY,
        "shared_state": server.get_global_state(),
        "federated_payload_policy": client_models[0].federated_payload_policy(),
        "shared_lora_factor_role": client_models[0].shared_lora_factor_role(),
        "local_lora_factor_role": local_factor_role,
        "local_personalized_states": local_personalized_states,
        "local_val": copy.deepcopy(list(local_val)),
        "local_val_summary": copy.deepcopy(local_val_summary),
        "round_metrics": copy.deepcopy(list(round_metrics)),
        "trainer_progress": [
            {"client_id": client_id, "global_epoch": int(trainer.global_epoch),
             "global_step": int(trainer.global_step)}
            for client_id, trainer in enumerate(client_trainers)
        ],
        "experiment": _experiment_manifest(args),
        "compatibility": _checkpoint_compatibility_manifest(args),
        "architecture": client_models[0].architecture_manifest(),
        "frozen_base_sha256": _frozen_base_sha256(client_models[0]),
        "parameter_counts": client_models[0].count_params(),
    }


def _load_checkpoint(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported federated checkpoint format: {path}")
    try:
        schema_version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported federated checkpoint format: {path}") from error
    if schema_version not in SUPPORTED_FEDERATED_CHECKPOINT_SCHEMAS:
        raise ValueError(
            f"Unsupported federated checkpoint schema {schema_version} in {path}; "
            f"supported={SUPPORTED_FEDERATED_CHECKPOINT_SCHEMAS}"
        )
    if payload.get("checkpoint_kind") != "federated_personalized":
        raise ValueError(f"Not a personalized federated checkpoint: {path}")
    _cpu_clone_state(payload.get("shared_state"))
    return payload


def _checkpoint_local_personalized_states(
    payload: dict, args
) -> tuple[Optional[str], Optional[list]]:
    """Normalize schema-4 FedSA and schema-5 factor-personalized checkpoints."""
    schema_version = int(payload.get("schema_version", -1))
    if schema_version == 4:
        if args.fl_method == "fixed_share_b_lora":
            raise ValueError(
                "Fixed Share-B requires checkpoint schema 5 with client-local A states"
            )
        local_b_states = payload.get("local_B_states")
        if args.fl_method == "fedsa_lora":
            return "B", local_b_states
        if local_b_states is not None:
            raise ValueError("Only schema-4 FedSA checkpoints may contain local_B_states")
        return None, None

    if schema_version != FEDERATED_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported checkpoint schema: {schema_version}")
    role = payload.get("local_lora_factor_role")
    states = payload.get("local_personalized_states")
    if role not in (None, "A", "B"):
        raise ValueError(f"Invalid checkpoint local LoRA factor role: {role!r}")
    return role, states


def _apply_checkpoint(
    payload: dict, client_models: Sequence[RTDETRLoRA], args, data_info: dict
):
    checks = {
        "fl_method": (payload.get("fl_method"), args.fl_method),
        "num_clients": (int(payload.get("num_clients", -1)), int(args.num_clients)),
        "class_names": (list(payload.get("class_names", [])), list(data_info["class_names"])),
        "split_manifest_sha256": (
            str(payload.get("split_manifest_sha256", "")).lower(),
            str(data_info["split_manifest_sha256"]).lower(),
        ),
        "client_sample_counts": (
            list(payload.get("client_sample_counts", [])),
            [int(value) for value in data_info["client_sizes"]],
        ),
        "nonfloating_state_policy": (
            payload.get("nonfloating_state_policy"), NONFLOATING_STATE_POLICY
        ),
    }
    mismatches = [
        f"{key}: checkpoint={left!r}, current={right!r}"
        for key, (left, right) in checks.items() if left != right
    ]
    if mismatches:
        raise ValueError("Federated checkpoint mismatch: " + "; ".join(mismatches))

    saved_compatibility = payload.get("compatibility")
    current_compatibility = _checkpoint_compatibility_manifest(args)
    if not isinstance(saved_compatibility, dict):
        raise ValueError("Federated checkpoint has no strict compatibility manifest")
    compatibility_mismatches = [
        f"{key}: checkpoint={saved_compatibility.get(key)!r}, current={value!r}"
        for key, value in current_compatibility.items()
        if saved_compatibility.get(key) != value
    ]
    if compatibility_mismatches:
        raise ValueError(
            "Federated checkpoint configuration mismatch: "
            + "; ".join(compatibility_mismatches)
        )

    saved_base_digest = payload.get("frozen_base_sha256")
    current_base_digest = _frozen_base_sha256(client_models[0], refresh=True)
    if saved_base_digest != current_base_digest:
        raise ValueError(
            "Frozen pretrained base differs from the one used by the checkpoint: "
            f"checkpoint={saved_base_digest!r}, current={current_base_digest!r}"
        )

    if int(payload.get("schema_version", -1)) >= 5:
        expected_policy = client_models[0].federated_payload_policy()
        expected_shared_role = client_models[0].shared_lora_factor_role()
        policy_checks = {
            "federated_payload_policy": (
                payload.get("federated_payload_policy"), expected_policy
            ),
            "shared_lora_factor_role": (
                payload.get("shared_lora_factor_role"), expected_shared_role
            ),
        }
        policy_mismatches = [
            f"{key}: checkpoint={left!r}, current={right!r}"
            for key, (left, right) in policy_checks.items() if left != right
        ]
        if policy_mismatches:
            raise ValueError(
                "Federated checkpoint factor-sharing policy mismatch: "
                + "; ".join(policy_mismatches)
            )

    shared_state = payload["shared_state"]
    checkpoint_role, local_states = _checkpoint_local_personalized_states(payload, args)
    expected_role = client_models[0].local_personalized_factor_role()
    if checkpoint_role != expected_role:
        raise ValueError(
            "Checkpoint local LoRA factor role mismatch: "
            f"checkpoint={checkpoint_role!r}, expected={expected_role!r}"
        )
    if expected_role is not None:
        if not isinstance(local_states, list) or len(local_states) != args.num_clients:
            raise ValueError(
                f"{args.fl_method} checkpoint does not contain one local "
                f"{expected_role} state per client"
            )
        for client_id, state in enumerate(local_states):
            try:
                _cpu_clone_state(state)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid client {client_id} local {expected_role} state"
                ) from error
    elif local_states is not None:
        raise ValueError(
            f"{args.fl_method} checkpoint unexpectedly contains personalized states"
        )
    for client_id, model in enumerate(client_models):
        if model.local_personalized_factor_role() != expected_role:
            raise RuntimeError(
                f"Client {client_id} local LoRA factor role differs from client 0"
            )
        model.set_aggregation_state(shared_state)
        if expected_role is not None:
            model.set_local_personalized_state(local_states[client_id])


def _evaluate_client_local(
    client_models: Sequence[RTDETRLoRA], data_info: dict, args, split: str
) -> tuple[List[dict], dict]:
    entries, metrics = [], []
    for client_id, model in enumerate(client_models):
        result = evaluate_model(model, data_info["client_yamls"][client_id], args, split=split)
        print_metrics(f"  Client {client_id} local {split}", result)
        entries.append({
            "client_id": client_id,
            "num_images": int(data_info["client_eval_sizes"][client_id][split]),
            **result,
            "metrics": result,
        })
        metrics.append(result)
    sample_counts = [
        int(data_info["client_eval_sizes"][client_id][split])
        for client_id in range(len(client_models))
    ]
    return entries, _metric_summary(metrics, sample_counts=sample_counts)


def _evaluate_cross_client_matrix(
    client_models: Sequence[RTDETRLoRA], data_info: dict, args,
    local_test_entries: Sequence[dict],
) -> dict:
    size = len(client_models)
    matrix = [[None for _ in range(size)] for _ in range(size)]
    for model_client_id, model in enumerate(client_models):
        for test_client_id in range(size):
            if model_client_id == test_client_id:
                metrics = local_test_entries[model_client_id]["metrics"]
            else:
                metrics = evaluate_model(
                    model, data_info["client_yamls"][test_client_id], args, split="test"
                )
            matrix[model_client_id][test_client_id] = metrics
    return {
        "enabled": True,
        "row_semantics": "personalized_model_client_id",
        "column_semantics": "client_local_test_partition_id",
        "metrics": matrix,
        "AP": [[float(item["AP"]) for item in row] for row in matrix],
        "AP50": [[float(item["AP50"]) for item in row] for row in matrix],
        "AP75": [[float(item["AP75"]) for item in row] for row in matrix],
    }


def _evaluate_common_test(
    client_models: Sequence[RTDETRLoRA], data_info: dict, args
) -> dict:
    entries, metrics = [], []
    for client_id, model in enumerate(client_models):
        result = evaluate_model(model, data_info["full_yaml"], args, split="test")
        print_metrics(f"  Client {client_id} model on common pooled test", result)
        entries.append({"client_id": client_id, **result, "metrics": result})
        metrics.append(result)
    summary = _metric_summary(metrics)
    reference_metrics = {
        key: float(summary[key]["macro_mean"])
        for key in REPORT_METRICS
    }
    return {
        "enabled": True,
        "dataset_yaml": data_info["full_yaml"],
        "protocol": "each selected personalized model evaluated on the identical pooled test set",
        "per_client_model": entries,
        "macro": summary,
        "summary_across_personalized_models": summary,
        "reference_metrics": reference_metrics,
        # Direct AP fields are the across-personalized-model macro and are kept
        # for publication-table aggregation alongside non-personalized baselines.
        **reference_metrics,
    }


def _evaluate_mia(
    client_models: Sequence[RTDETRLoRA], data_info: dict, args
) -> dict:
    per_client = []
    for client_id, model in enumerate(client_models):
        print(f"  [MIA] Client {client_id}: extracting matched detection losses")
        member_losses = get_mia_losses(
            model, data_info["client_yamls"][client_id], args, split="train",
            max_samples=args.mia_max_samples,
        )
        nonmember_losses = get_mia_losses(
            model, data_info["client_yamls"][client_id], args, split="test",
            max_samples=args.mia_max_samples,
            exclude_file_names=data_info["client_mia_excluded_test_files"][client_id],
        )
        metrics = compute_mia_metrics(
            member_losses, nonmember_losses,
            calibration_fraction=args.mia_calibration_fraction,
            seed=client_attack_seed(args.seed, client_id),
        )
        metrics["source_disjoint_sampling"] = data_info[
            "client_mia_source_audit"
        ][client_id]
        per_client.append({"client_id": client_id, **metrics, "metrics": metrics})
        print(
            f"    AUC={metrics['auc_roc']:.4f}, "
            f"TPR@1%FPR={metrics['tpr_at_1fpr']:.4f}, ASR={metrics['asr']:.4f}"
        )
    summary = _mia_summary(per_client)
    return {
        "threat_model": "white_box_final_personalized_model_image_level_loss_attack",
        "member_split": "client_local_train",
        "nonmember_split": "client_local_test",
        "configuration": {
            "attack": "image_level_ground_truth_matched_detection_loss_threshold",
            "max_samples_per_membership_class_per_client": int(args.mia_max_samples),
            "calibration_fraction": float(args.mia_calibration_fraction),
            "nonmember_source_policy": (
                "exclude_test_source_components_present_in_any_train_client"
            ),
            "bootstrap": {
                "method": "stratified_nonparametric_full_attack_pipeline_percentile",
                "resamples": 1000,
                "confidence_level": 0.95,
                "recalibrates_direction_and_threshold": True,
            },
        },
        "per_client": per_client,
        "macro": summary,
        "summary": summary,
        "auc_roc": summary["auc_roc"]["macro_mean"],
        "tpr_at_1fpr": summary["tpr_at_1fpr"]["macro_mean"],
        "asr": summary["asr"]["macro_mean"],
    }


def _final_evaluation(
    client_models: Sequence[RTDETRLoRA], data_info: dict, args
) -> dict:
    print("\n" + "=" * 60)
    print("Final evaluation of the validation-selected personalized checkpoint")
    print("=" * 60)
    local_test, client_summary = _evaluate_client_local(
        client_models, data_info, args, split="test"
    )
    common_test = _evaluate_common_test(client_models, data_info, args)
    cross_client = (
        _evaluate_cross_client_matrix(client_models, data_info, args, local_test)
        if getattr(args, "cross_client_eval", True)
        else None
    )
    mia = _evaluate_mia(client_models, data_info, args) if args.run_mia else None
    return {
        "client_local_test": local_test,
        "client_summary": client_summary,
        "common_test": common_test,
        "cross_client_matrix": cross_client,
        "mia": mia,
    }


def _build_final_results(
    *, args, data_info: dict, client_models: Sequence[RTDETRLoRA],
    selected_checkpoint: str, selected_payload: dict, round_metrics: Sequence[dict],
    rounds_executed: int,
) -> dict:
    parameter_counts = client_models[0].count_params()
    shared_state = selected_payload["shared_state"]
    communication = _communication_accounting(shared_state, args.num_clients, rounds_executed)
    if communication["one_client_one_way"]["params"] != parameter_counts["communication_params"]:
        raise RuntimeError("Model parameter count and actual federated payload disagree")
    if communication["one_client_one_way"]["bytes"] != parameter_counts["communication_bytes"]:
        raise RuntimeError("Model byte count and actual federated payload disagree")
    full_reference_state = {
        key: value
        for key, value in client_models[0].model.state_dict().items()
        if not key.endswith(".lora_A") and not key.endswith(".lora_B")
    }
    full_reference = _payload_statistics(full_reference_state)
    if full_reference["params"] != parameter_counts["full_ft_payload_params"]:
        raise RuntimeError("Full-FT reference payload and model parameter count disagree")
    byte_ratio = (
        100.0 * communication["one_client_one_way"]["bytes"] / full_reference["bytes"]
    )
    element_ratio = (
        100.0 * communication["one_client_one_way"]["params"] / full_reference["params"]
    )
    if not np.isclose(
        element_ratio, parameter_counts["communication_ratio_pct"], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Element-count communication ratio disagrees with model accounting")
    communication["full_ft_one_client_one_way_reference"] = full_reference
    communication["byte_ratio_vs_full_ft_pct"] = float(byte_ratio)
    communication["byte_saving_vs_full_ft_pct"] = float(100.0 - byte_ratio)
    communication["element_ratio_vs_full_ft_pct"] = float(element_ratio)
    communication["element_saving_vs_full_ft_pct"] = float(100.0 - element_ratio)
    # Backward-compatible aliases are explicitly byte-based.
    communication["communication_ratio_vs_full_ft_pct"] = float(byte_ratio)
    communication["saving_vs_full_ft_pct"] = float(100.0 - byte_ratio)
    communication["primary_efficiency_definition"] = "byte_saving_vs_full_ft_pct"

    evaluation = _final_evaluation(client_models, data_info, args)
    client_summary = evaluation["client_summary"]
    uses_lora = args.fl_method != "full_ft"
    return {
        "result_schema_version": FEDERATED_RESULT_SCHEMA,
        "status": "complete",
        "schema_version": FEDERATED_RESULT_SCHEMA,
        "federated_checkpoint_schema_version": int(
            selected_payload.get("schema_version", FEDERATED_CHECKPOINT_SCHEMA)
        ),
        "mode": "fl",
        "method": f"FL+{args.fl_method}",
        "fl_method": args.fl_method,
        "seed": int(args.seed),
        "partition_seed": int(args.partition_seed),
        "partition": args.partition,
        "dirichlet_alpha": (
            float(args.dirichlet_alpha) if args.partition == "dirichlet" else None
        ),
        "lora_rank": int(args.lora_rank) if uses_lora else None,
        "lora_alpha": float(args.lora_alpha) if uses_lora else None,
        "lora_scaling": (
            float(args.lora_alpha / args.lora_rank) if uses_lora else None
        ),
        "apply_lora_backbone": bool(args.apply_lora_backbone) if uses_lora else None,
        "apply_lora_decoder": bool(args.apply_lora_decoder) if uses_lora else None,
        "backbone_min_channels": int(args.backbone_min_channels) if uses_lora else None,
        "num_clients": int(args.num_clients),
        "rounds_planned": int(args.fl_rounds),
        "rounds_executed": int(rounds_executed),
        "local_epochs": int(args.local_epochs),
        "selection": {
            "criterion": str(selected_payload["selection"]),
            "checkpoint": selected_checkpoint,
            "round": int(selected_payload["round"]),
        },
        "nonfloating_state_policy": NONFLOATING_STATE_POLICY,
        "federated_payload_policy": client_models[0].federated_payload_policy(),
        "shared_lora_factor_role": client_models[0].shared_lora_factor_role(),
        "client_local_lora_factor_role": (
            client_models[0].local_personalized_factor_role()
        ),
        "client_local_val": copy.deepcopy(selected_payload["local_val"]),
        "client_val_summary": copy.deepcopy(selected_payload["local_val_summary"]),
        "round_metrics": list(round_metrics),
        "client_local_test": evaluation["client_local_test"],
        "client_summary": client_summary,
        "common_test": evaluation["common_test"],
        "cross_client_matrix": evaluation["cross_client_matrix"],
        "mia": evaluation["mia"],
        "communication": communication,
        "parameter_counts": parameter_counts,
        "architecture": client_models[0].architecture_manifest(),
        "experiment": _experiment_manifest(args),
        "training_experiment": copy.deepcopy(selected_payload["experiment"]),
        "split_manifest_sha256": str(data_info["split_manifest_sha256"]).lower(),
        "split_metadata": data_info.get("split_metadata", {}),
        "final_client_test": [entry["metrics"] for entry in evaluation["client_local_test"]],
        "avg_test_AP": client_summary["AP"]["macro_mean"],
        "avg_test_AP50": client_summary["AP50"]["macro_mean"],
        "avg_test_AP75": client_summary["AP75"]["macro_mean"],
        "std_test_AP": client_summary["AP"]["sample_std"],
        "std_test_AP50": client_summary["AP50"]["sample_std"],
        "std_test_AP75": client_summary["AP75"]["sample_std"],
        "total_params": parameter_counts["total_params"],
        "trainable_params": parameter_counts["trainable_params"],
        "trainable_params_m": parameter_counts["trainable_params"] / 1_000_000.0,
        "comm_params": communication["one_client_one_way"]["params"],
        "comm_params_m": communication["one_client_one_way"]["params"] / 1_000_000.0,
        "param_efficiency": parameter_counts["parameter_saving_pct"],
        "comm_efficiency": communication["byte_saving_vs_full_ft_pct"],
        "comm_efficiency_definition": "byte_saving_vs_full_ft_pct",
        "comm_byte_saving_pct": communication["byte_saving_vs_full_ft_pct"],
        "comm_element_saving_pct": communication["element_saving_vs_full_ft_pct"],
        "one_client_one_way_mb": communication["one_client_one_way"]["mb"],
        "per_round_comm_mb": communication["round_total"]["mb"],
        "total_comm_mb": communication["cumulative_total"]["mb"],
    }


def _save_plots(results: dict, data_info: dict, args):
    round_metrics = results.get("round_metrics", [])
    if round_metrics:
        plot_fl_metrics(
            round_metrics, os.path.join(args.exp_dir, "fl_training_curves.png"),
            title=f"FL Training: {args.fl_method} ({args.partition})",
        )
        loss_data = {
            f"Client {client_id}": [record["client_losses"][client_id] for record in round_metrics]
            for client_id in range(args.num_clients)
        }
        plot_training_loss(
            loss_data, os.path.join(args.exp_dir, "fl_loss_curves.png"),
            title=f"Per-round local loss: {args.fl_method}",
            xlabel="Federated round",
        )
        component_series = {}
        learning_rate_series = {}
        for client_id in range(args.num_clients):
            component_names = sorted({
                component
                for round_record in round_metrics
                for epoch_components in round_record["client_train"][client_id]["loss_components"]
                for component in epoch_components
            })
            flattened_components = [
                epoch_components
                for round_record in round_metrics
                for epoch_components in round_record["client_train"][client_id]["loss_components"]
            ]
            for component in component_names:
                component_series[f"Client {client_id} {component}"] = [
                    float(epoch.get(component, float("nan")))
                    for epoch in flattened_components
                ]
            flattened_lrs = [
                epoch_lrs
                for round_record in round_metrics
                for epoch_lrs in round_record["client_train"][client_id]["learning_rates"]
            ]
            if flattened_lrs:
                learning_rate_series[f"Client {client_id} max LR"] = [
                    max(map(float, values)) for values in flattened_lrs
                ]
                learning_rate_series[f"Client {client_id} min LR"] = [
                    min(map(float, values)) for values in flattened_lrs
                ]
        if component_series:
            plot_training_loss(
                component_series,
                os.path.join(args.exp_dir, "fl_loss_components.png"),
                title=f"Effective-epoch RT-DETR loss components: {args.fl_method}",
                xlabel="Effective local epoch",
                ylabel="Loss component",
            )
        if learning_rate_series:
            plot_training_loss(
                learning_rate_series,
                os.path.join(args.exp_dir, "fl_learning_rate_schedule.png"),
                title=f"Effective-epoch warmup-cosine schedule: {args.fl_method}",
                xlabel="Effective local epoch",
                ylabel="Learning rate",
            )
    plot_client_data_distribution(
        data_info["client_splits"], data_info["train_coco"],
        os.path.join(args.exp_dir, "client_data_distribution.png"),
    )


def run_federated_learning(args, data_info: dict) -> dict:
    """Train all clients and test the validation-selected personalized models."""
    _validate_data_info(args, data_info)
    print("\n" + "=" * 60)
    print(
        f"Federated Learning: {args.fl_method} | clients={args.num_clients} | "
        f"rounds={args.fl_rounds} | local_epochs={args.local_epochs}"
    )
    print(f"Non-floating state policy: {NONFLOATING_STATE_POLICY}")
    print("=" * 60)

    client_models = _new_client_models(args, data_info)
    client_trainers = [
        FLLocalTrainer(
            model, data_info["client_yamls"][client_id], args, client_id=client_id,
            total_epochs=args.fl_rounds * args.local_epochs,
        )
        for client_id, model in enumerate(client_models)
    ]
    server = FLServer(args.fl_method, args.num_clients)
    server.initialize_global_state(client_models[0].get_aggregation_state())
    initial_payload = _payload_statistics(server.get_global_state())
    print(
        f"[FL] One-client one-way payload: {initial_payload['params']/1e6:.4f}M values, "
        f"{initial_payload['mb']:.3f} MB"
    )

    weights_dir = os.path.join(args.exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_path = os.path.join(weights_dir, "best_federated.pt")
    last_path = os.path.join(weights_dir, "last_federated.pt")
    round_metrics = []
    best_val_ap = -float("inf")
    best_round = 0
    no_improvement = 0

    for round_index in range(args.fl_rounds):
        round_number = round_index + 1
        print(f"\n{'=' * 60}\nFL Round {round_number}/{args.fl_rounds}\n{'=' * 60}")
        shared_state = server.get_global_state()
        for model in client_models:
            model.set_aggregation_state(shared_state)
        for trainer in client_trainers:
            trainer.set_global_params()

        client_train = []
        for client_id, trainer in enumerate(client_trainers):
            print(f"\n--- Client {client_id}: {args.local_epochs} local epochs ---")
            client_train.append(trainer.train_epoch(epochs=args.local_epochs))

        client_states = [model.get_aggregation_state() for model in client_models]
        aggregated_state = server.aggregate(client_states, data_info["client_sizes"])
        for model in client_models:
            model.set_aggregation_state(aggregated_state)

        local_val, local_val_summary = _evaluate_client_local(
            client_models, data_info, args, split="val"
        )
        round_communication = _communication_accounting(
            aggregated_state, args.num_clients, round_number
        )
        local_val_metrics = [entry["metrics"] for entry in local_val]
        round_result = {
            "round": round_number,
            "client_train": client_train,
            "client_losses": [record["avg_loss"] for record in client_train],
            "client_local_val": local_val,
            "client_eval": local_val_metrics,
            "client_summary": local_val_summary,
            "avg_AP": local_val_summary["AP"]["macro_mean"],
            "avg_AP50": local_val_summary["AP50"]["macro_mean"],
            "avg_AP75": local_val_summary["AP75"]["macro_mean"],
            "std_AP": local_val_summary["AP"]["sample_std"],
            "std_AP50": local_val_summary["AP50"]["sample_std"],
            "std_AP75": local_val_summary["AP75"]["sample_std"],
            "communication": round_communication,
            "comm_cost": initial_payload,
            "normalized_fedavg_weights": list(server.last_normalized_weights),
        }
        round_metrics.append(round_result)
        print(
            f"[Round {round_number}] macro local-val AP="
            f"{round_result['avg_AP']:.4f} ± {round_result['std_AP']:.4f} (sample SD)"
        )

        if round_result["avg_AP"] > best_val_ap:
            best_val_ap = round_result["avg_AP"]
            best_round = round_number
            no_improvement = 0
            checkpoint = _make_checkpoint(
                args=args, data_info=data_info, client_models=client_models, server=server,
                client_trainers=client_trainers, round_number=round_number,
                selection="best_macro_client_local_val_AP", local_val=local_val,
                local_val_summary=local_val_summary, round_metrics=round_metrics,
                training_rounds_executed=round_number,
            )
            _atomic_torch_save(checkpoint, best_path)
            print(f"[Checkpoint] best personalized state updated at round {round_number}")
        else:
            no_improvement += 1

        if args.visualize_interval > 0 and round_number % args.visualize_interval == 0:
            for client_id, model in enumerate(client_models):
                save_detection_samples(
                    model, data_info["client_yamls"][client_id], args,
                    os.path.join(args.exp_dir, "detection_vis"),
                    tag=f"round_{round_number:03d}/client_{client_id}",
                    num_samples=args.vis_samples, seed=args.seed, splits=("val",),
                )

        if args.patience > 0 and no_improvement >= args.patience:
            print(f"[EarlyStop] No macro local-val AP improvement for {args.patience} rounds")
            break

    rounds_executed = len(round_metrics)
    last_checkpoint = _make_checkpoint(
        args=args, data_info=data_info, client_models=client_models, server=server,
        client_trainers=client_trainers, round_number=rounds_executed,
        selection="last_executed_round", local_val=round_metrics[-1]["client_local_val"],
        local_val_summary=round_metrics[-1]["client_summary"], round_metrics=round_metrics,
        training_rounds_executed=rounds_executed,
    )
    last_checkpoint["best_round"] = int(best_round)
    last_checkpoint["best_val_ap"] = float(best_val_ap)
    _atomic_torch_save(last_checkpoint, last_path)

    if not os.path.isfile(best_path):
        raise RuntimeError("Federated training completed without a best checkpoint")
    best_checkpoint = _load_checkpoint(best_path)
    best_checkpoint["training_rounds_executed"] = rounds_executed
    best_checkpoint["round_metrics"] = copy.deepcopy(round_metrics)
    best_checkpoint["best_round"] = int(best_round)
    best_checkpoint["best_val_ap"] = float(best_val_ap)
    _atomic_torch_save(best_checkpoint, best_path)
    _apply_checkpoint(best_checkpoint, client_models, args, data_info)

    results = _build_final_results(
        args=args, data_info=data_info, client_models=client_models,
        selected_checkpoint=best_path, selected_payload=best_checkpoint,
        round_metrics=round_metrics, rounds_executed=rounds_executed,
    )
    _save_plots(results, data_info, args)
    save_results_json(results, os.path.join(args.exp_dir, "fl_results.json"))
    return results


def evaluate_federated_checkpoint(
    args, data_info: dict, exp_dir: str, preference: str = "best"
) -> dict:
    """Restore every personalized client from a best/last FL bundle and evaluate."""
    _validate_data_info(args, data_info)
    if preference not in ("best", "last"):
        raise ValueError("preference must be 'best' or 'last'")
    weights_dir = os.path.join(os.path.abspath(exp_dir), "weights")
    preferred = os.path.join(weights_dir, f"{preference}_federated.pt")
    fallback_name = "last" if preference == "best" else "best"
    fallback = os.path.join(weights_dir, f"{fallback_name}_federated.pt")
    checkpoint_path = preferred if os.path.isfile(preferred) else fallback
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No personalized federated checkpoint found in {weights_dir}")

    payload = _load_checkpoint(checkpoint_path)
    client_models = _new_client_models(args, data_info)
    _apply_checkpoint(payload, client_models, args, data_info)
    rounds_executed = int(payload.get("training_rounds_executed", payload["round"]))
    results = _build_final_results(
        args=args, data_info=data_info, client_models=client_models,
        selected_checkpoint=checkpoint_path, selected_payload=payload,
        round_metrics=payload.get("round_metrics", []), rounds_executed=rounds_executed,
    )
    results["resumed_from"] = checkpoint_path
    save_results_json(results, os.path.join(os.path.abspath(exp_dir), "fl_results.json"))
    return results
