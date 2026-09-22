#!/usr/bin/env python3
"""Reproducible entry point for the AOD-4 RT-DETR federated study."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import ultralytics

from configs.config import get_args
from data.dataset import prepare_data
from models.rtdetr_lora import RTDETRLoRA
from trainers.fl_server import run_federated_learning
from trainers.trainer import (
    evaluate_model,
    get_mia_losses,
    print_metrics,
    rtdetr_augmentation_manifest,
    train_with_ultralytics,
)
from utils.visualization import (
    plot_client_data_distribution,
    plot_training_loss,
    save_results_json,
)


RESULT_SCHEMA_VERSION = 2
PRIMARY_METRICS = ("AP", "AP50", "AP75")
MIA_ATTACK_NAME = "image_level_ground_truth_matched_detection_loss_threshold"
MIA_NONMEMBER_SOURCE_POLICY = (
    "exclude_test_source_components_present_in_any_train_client"
)
MIA_BOOTSTRAP_SPEC = {
    "method": "stratified_nonparametric_full_attack_pipeline_percentile",
    "resamples": 1000,
    "confidence_level": 0.95,
    "recalibrates_direction_and_threshold": True,
}


class TeeOutput:
    """Mirror stdout/stderr to a UTF-8 experiment log."""

    def __init__(self, log_path: str, stream):
        self.stream = stream
        self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


def set_seed(seed: int):
    """Set host and accelerator RNGs; deterministic kernels are requested where practical."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_existing_result(exp_dir: str, filename: str) -> dict:
    path = os.path.join(exp_dir, filename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _runtime_arguments(args) -> dict:
    """Return semantically normalized CLI provenance for the run manifest."""
    arguments = dict(vars(args))
    if args.partition != "dirichlet":
        # ``dirichlet_alpha`` is only an argparse default for an explicit IID
        # partition.  Preserve one unambiguous representation in every
        # manifest layer rather than recording an unused alpha=0.4 here.
        arguments["dirichlet_alpha"] = None
    return arguments


def _runtime_manifest(args, data_info: dict) -> dict:
    local_weight = args.model_weights if args.model_weights else None
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "command": sys.argv,
        "arguments": _runtime_arguments(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "ultralytics": ultralytics.__version__,
        "model_weight_identifier": local_weight or f"{args.model_name}.pt",
        "model_weight_sha256": _sha256(local_weight),
        "split_file_sha256": _sha256(args.split_file),
        "split_metadata": data_info["split_metadata"],
        "reproducibility_note": (
            "Deterministic algorithms are requested with warn_only=True; CUDA kernels may still "
            "emit a warning when no deterministic implementation exists."
        ),
    }


def summarize_client_metrics(metrics: list[dict], sample_counts: Optional[Iterable[int]] = None) -> dict:
    """Separate client-macro disparity from sample-weighted performance."""
    if not metrics:
        raise ValueError("Cannot summarize an empty client metric list")
    if sample_counts is None:
        weights = np.full(len(metrics), 1.0 / len(metrics), dtype=np.float64)
    else:
        counts = np.asarray(list(sample_counts), dtype=np.float64)
        if len(counts) != len(metrics) or np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError(f"Invalid client sample counts: {counts}")
        weights = counts / counts.sum()

    summary = {
        "definition": (
            "macro_mean is the unweighted client mean; client_sample_sd uses ddof=1; "
            "weighted_mean uses client test image counts. Pooled AP is reported separately."
        )
    }
    for key in PRIMARY_METRICS:
        values = np.asarray([float(item[key]) for item in metrics], dtype=np.float64)
        summary[key] = {
            "macro_mean": float(values.mean()),
            "client_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "weighted_mean": float(np.dot(weights, values)),
            "worst_client": float(values.min()),
            "best_client": float(values.max()),
            "max_min_gap": float(values.max() - values.min()),
        }
    return summary


def _evaluate_client_partitions(model, data_info: dict, args) -> list[dict]:
    results = []
    for client_id, data_yaml in enumerate(data_info["client_yamls"]):
        metrics = evaluate_model(model, data_yaml, args, split="test")
        print_metrics(f"Client {client_id} local test", metrics)
        results.append(metrics)
    return results


def _summarize_mia(per_client: list[dict], args) -> dict:
    keys = ("auc_roc", "tpr_at_1fpr", "asr")
    summary = {}
    for key in keys:
        values = np.asarray([float(item[key]) for item in per_client], dtype=np.float64)
        summary[key] = {
            "macro_mean": float(values.mean()),
            "client_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {
        "threat_model": "white-box final-model ground-truth RT-DETR detection-loss attack",
        "interpretation": "Empirical privacy-leakage audit; not a confidentiality guarantee.",
        "configuration": {
            "attack": MIA_ATTACK_NAME,
            "max_samples_per_membership_class_per_client": int(args.mia_max_samples),
            "calibration_fraction": float(args.mia_calibration_fraction),
            "nonmember_source_policy": MIA_NONMEMBER_SOURCE_POLICY,
            "bootstrap": dict(MIA_BOOTSTRAP_SPEC),
        },
        "per_client": per_client,
        "macro": summary,
    }


def _run_mia_for_client_models(models: list, data_info: dict, args) -> dict:
    from utils.mia import client_attack_seed, compute_mia_metrics

    if len(models) not in (1, args.num_clients):
        raise ValueError("MIA needs one shared model or one personalized model per client")
    per_client = []
    for client_id, data_yaml in enumerate(data_info["client_yamls"]):
        model = models[0] if len(models) == 1 else models[client_id]
        member = get_mia_losses(
            model, data_yaml, args, split="train", max_samples=args.mia_max_samples
        )
        nonmember = get_mia_losses(
            model,
            data_yaml,
            args,
            split="test",
            max_samples=args.mia_max_samples,
            exclude_file_names=data_info["client_mia_excluded_test_files"][client_id],
        )
        metrics = compute_mia_metrics(
            member,
            nonmember,
            seed=client_attack_seed(args.seed, client_id),
            calibration_fraction=args.mia_calibration_fraction,
        )
        metrics["client_id"] = client_id
        metrics["source_disjoint_sampling"] = data_info[
            "client_mia_source_audit"
        ][client_id]
        per_client.append(metrics)
        print(
            f"[MIA] client={client_id} AUC={metrics['auc_roc']:.4f}, "
            f"TPR@1%FPR={metrics['tpr_at_1fpr']:.4f}, ASR={metrics['asr']:.4f}"
        )
    return _summarize_mia(per_client, args)


def _training_experiment_manifest(args) -> dict:
    keys = (
        "mode", "fl_method", "model_name", "model_weights", "num_classes",
        "num_clients", "partition", "dirichlet_alpha", "lora_rank", "lora_alpha",
        "lora_dropout", "apply_lora_backbone", "apply_lora_decoder",
        "backbone_min_channels", "fl_rounds", "local_epochs", "centralized_epochs",
        "solo_epochs", "batch_size", "img_size", "lr", "head_lr",
        "backbone_lr_ratio", "weight_decay", "warmup_epochs", "min_lr_ratio",
        "grad_clip_norm", "close_mosaic_epochs", "fedprox_mu",
        "reset_optimizer_each_round", "amp",
        "patience", "val_interval", "seed", "partition_seed", "num_workers",
        "cross_client_eval", "visualize_interval", "vis_samples",
        "mia_max_samples", "mia_calibration_fraction",
    )
    manifest = {key: getattr(args, key, None) for key in keys}
    if args.partition != "dirichlet":
        # Alpha has no statistical meaning for the explicit label-blind IID
        # partition.  Recording the argparse default here would create a false
        # protocol difference and make the manifest scientifically ambiguous.
        manifest["dirichlet_alpha"] = None
    manifest.update({
        "optimizer": "AdamW",
        "lr_schedule": "global_step_linear_warmup_then_cosine_decay",
        "mia_attack": MIA_ATTACK_NAME,
        "mia_nonmember_source_policy": MIA_NONMEMBER_SOURCE_POLICY,
        "mia_bootstrap": dict(MIA_BOOTSTRAP_SPEC),
        "augmentation_protocol": rtdetr_augmentation_manifest(args),
        "effective_close_mosaic_epochs": int(args.close_mosaic_epochs),
        "checkpoint_selection": (
            "single_client_validation_AP"
            if args.mode == "solo"
            else "macro_client_local_validation_AP"
        ),
    })
    return manifest


def _base_result(args, data_info: dict, model: RTDETRLoRA) -> dict:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "mode": args.mode,
        "fl_method": args.fl_method,
        "seed": args.seed,
        "partition_seed": args.partition_seed,
        "partition": args.partition,
        "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
        "num_clients": args.num_clients,
        "lora_rank": args.lora_rank if args.fl_method != "full_ft" else None,
        "lora_alpha": args.lora_alpha if args.fl_method != "full_ft" else None,
        "lora_scaling": (
            args.lora_alpha / args.lora_rank if args.fl_method != "full_ft" else None
        ),
        "apply_lora_backbone": (
            args.apply_lora_backbone if args.fl_method != "full_ft" else None
        ),
        "apply_lora_decoder": (
            args.apply_lora_decoder if args.fl_method != "full_ft" else None
        ),
        "parameter_counts": model.count_params(),
        "architecture": model.architecture_manifest(),
        "training_experiment": _training_experiment_manifest(args),
        "split_file": os.path.abspath(args.split_file),
        "split_file_sha256": _sha256(args.split_file),
        "split_metadata": data_info.get("split_metadata", {}),
    }


def _load_standalone_checkpoint(model: RTDETRLoRA, exp_dir: str) -> tuple[str, dict]:
    candidates = (
        os.path.join(exp_dir, "weights", "best_full.pt"),
        os.path.join(exp_dir, "weights", "last_full.pt"),
    )
    checkpoint = next((path for path in candidates if os.path.isfile(path)), None)
    if checkpoint is None:
        raise FileNotFoundError(f"No standalone checkpoint found under {exp_dir}/weights")
    extra = model.load_model(checkpoint)
    return checkpoint, extra


def _validate_standalone_resume_result(
    previous: dict, args, data_info: dict, expected_mode: str,
    expected_client_id: Optional[int] = None,
):
    """Prevent evaluating a checkpoint under a different split/model protocol."""
    if not previous:
        raise FileNotFoundError(
            f"Cannot validate --resume: the prior {expected_mode} result JSON is missing"
        )
    checks = {
        "mode": (previous.get("mode"), expected_mode),
        "fl_method": (previous.get("fl_method"), args.fl_method),
        "seed": (int(previous.get("seed", -1)), int(args.seed)),
        "partition_seed": (
            int(previous.get("partition_seed", -1)), int(args.partition_seed)
        ),
        "num_clients": (int(previous.get("num_clients", -1)), int(args.num_clients)),
        "partition": (previous.get("partition"), args.partition),
        "split_file_sha256": (
            str(previous.get("split_file_sha256", "")).lower(),
            str(data_info["split_manifest_sha256"]).lower(),
        ),
    }
    if expected_client_id is not None:
        checks["client_id"] = (
            int(previous.get("client_id", -1)), int(expected_client_id)
        )
    if args.partition == "dirichlet":
        checks["dirichlet_alpha"] = (
            float(previous.get("dirichlet_alpha", float("nan"))),
            float(args.dirichlet_alpha),
        )
    if args.fl_method != "full_ft":
        checks.update({
            "lora_rank": (int(previous.get("lora_rank", -1)), int(args.lora_rank)),
            "lora_alpha": (
                float(previous.get("lora_alpha", float("nan"))), float(args.lora_alpha)
            ),
            "apply_lora_backbone": (
                previous.get("apply_lora_backbone"), bool(args.apply_lora_backbone)
            ),
            "apply_lora_decoder": (
                previous.get("apply_lora_decoder"), bool(args.apply_lora_decoder)
            ),
        })
    previous_training = previous.get("training_experiment")
    current_training = _training_experiment_manifest(args)
    if not isinstance(previous_training, dict):
        checks["training_experiment"] = (previous_training, "complete manifest")
    else:
        evaluation_only_keys = {
            "mia_max_samples", "mia_calibration_fraction", "mia_attack", "mia_bootstrap"
        }
        for key, current_value in current_training.items():
            if key not in evaluation_only_keys:
                checks[f"training_experiment.{key}"] = (
                    previous_training.get(key), current_value
                )
    mismatches = [
        f"{key}: result={left!r}, current={right!r}"
        for key, (left, right) in checks.items() if left != right
    ]
    if mismatches:
        raise ValueError("Standalone resume mismatch: " + "; ".join(mismatches))


def run_solo(args, data_info: dict) -> dict:
    client_id = int(args.client_id)
    if not 0 <= client_id < args.num_clients:
        raise ValueError(f"--client_id must be in [0, {args.num_clients - 1}]")
    model = RTDETRLoRA(args, data_info["class_names"])
    if args.resume:
        previous = _load_existing_result(args.exp_dir, "solo_results.json")
        _validate_standalone_resume_result(
            previous, args, data_info, expected_mode="solo", expected_client_id=client_id
        )
        checkpoint, checkpoint_extra = _load_standalone_checkpoint(model, args.exp_dir)
        training = previous.get("training", {})
        training = {
            **training,
            "evaluated_from": checkpoint,
            "checkpoint_extra": checkpoint_extra,
        }
    else:
        training = train_with_ultralytics(
            model,
            data_info["client_yamls"][client_id],
            args.solo_epochs,
            args,
            args.exp_dir,
            run_name=f"solo_client_{client_id}",
        )

    own_test = evaluate_model(
        model, data_info["client_yamls"][client_id], args, split="test"
    )
    print_metrics(f"Solo client {client_id} own test", own_test)
    common_test = evaluate_model(model, data_info["full_yaml"], args, split="test")
    print_metrics(f"Solo client {client_id} common pooled test", common_test)
    cross_client = _evaluate_client_partitions(model, data_info, args) if args.cross_client_eval else []

    results = _base_result(args, data_info, model)
    results.update({
        "method": f"Local-{args.fl_method}",
        "client_id": client_id,
        "epochs_budget": args.solo_epochs,
        "training": training,
        "own_client_test": own_test,
        "common_test": common_test,
        "cross_client_test": cross_client,
        "communication": {
            "applicable": False,
            "reason": "No inter-client model-update communication in isolated local training.",
        },
    })
    if cross_client:
        results["cross_client_summary"] = summarize_client_metrics(
            cross_client, [item["test"] for item in data_info["client_eval_sizes"]]
        )
    if args.run_mia:
        from utils.mia import client_attack_seed, compute_mia_metrics
        data_yaml = data_info["client_yamls"][client_id]
        member = get_mia_losses(model, data_yaml, args, "train", args.mia_max_samples)
        nonmember = get_mia_losses(
            model,
            data_yaml,
            args,
            "test",
            args.mia_max_samples,
            exclude_file_names=data_info["client_mia_excluded_test_files"][client_id],
        )
        mia = compute_mia_metrics(
            member,
            nonmember,
            seed=client_attack_seed(args.seed, client_id),
            calibration_fraction=args.mia_calibration_fraction,
        )
        mia["client_id"] = client_id
        mia["source_disjoint_sampling"] = data_info[
            "client_mia_source_audit"
        ][client_id]
        results["mia"] = _summarize_mia([mia], args)
    return results


def run_centralized(args, data_info: dict) -> dict:
    model = RTDETRLoRA(args, data_info["class_names"])
    if args.resume:
        previous = _load_existing_result(args.exp_dir, "centralized_results.json")
        _validate_standalone_resume_result(
            previous, args, data_info, expected_mode="centralized"
        )
        checkpoint, checkpoint_extra = _load_standalone_checkpoint(model, args.exp_dir)
        training = previous.get("training", {})
        training = {
            **training,
            "evaluated_from": checkpoint,
            "checkpoint_extra": checkpoint_extra,
        }
    else:
        training = train_with_ultralytics(
            model,
            data_info["full_yaml"],
            args.centralized_epochs,
            args,
            args.exp_dir,
            run_name="centralized",
            validation_yamls=data_info["client_yamls"],
        )

    client_local = _evaluate_client_partitions(model, data_info, args)
    common_test = evaluate_model(model, data_info["full_yaml"], args, split="test")
    print_metrics("Centralized common pooled test", common_test)
    summary = summarize_client_metrics(
        client_local, [item["test"] for item in data_info["client_eval_sizes"]]
    )
    results = _base_result(args, data_info, model)
    results.update({
        "method": f"Centralized-{args.fl_method}",
        "epochs_budget": args.centralized_epochs,
        "training": training,
        "client_local_test": client_local,
        "client_summary": summary,
        "common_test": common_test,
        "communication": {
            "applicable": False,
            "reason": (
                "FL model-update communication is not applicable. This baseline requires raw "
                "training data to be pooled centrally; that transfer is not conflated with FL payload."
            ),
        },
    })
    if args.run_mia:
        results["mia"] = _run_mia_for_client_models([model], data_info, args)
    return results


def _plot_standalone_history(results: dict, args):
    history = results.get("training", {}).get("history", [])
    if not history:
        return
    component_keys = sorted({
        key for row in history for key in row.get("loss_components", {})
    })
    loss_series = {"total_loss": [float(row["loss"]) for row in history]}
    for key in component_keys:
        loss_series[key] = [
            float(row.get("loss_components", {}).get(key, float("nan")))
            for row in history
        ]
    plot_training_loss(
        loss_series,
        os.path.join(args.exp_dir, "training_loss_components.png"),
        title=f"{results['method']} training losses",
    )
    lr_series = {}
    if all(row.get("learning_rates") for row in history):
        lr_series = {
            "maximum group LR": [max(map(float, row["learning_rates"])) for row in history],
            "minimum group LR": [min(map(float, row["learning_rates"])) for row in history],
        }
    if lr_series:
        plot_training_loss(
            lr_series,
            os.path.join(args.exp_dir, "learning_rate_schedule.png"),
            title=f"{results['method']} warmup-cosine schedule",
            xlabel="Effective epoch",
            ylabel="Learning rate",
        )


def main(argv=None):
    args = get_args(argv)
    if args.resume:
        args.exp_dir = os.path.abspath(args.resume)
        if not os.path.isdir(args.exp_dir):
            raise FileNotFoundError(f"--resume experiment directory not found: {args.exp_dir}")

    set_seed(args.seed)
    data_info = prepare_data(args)
    manifest_name = "evaluation_manifest.json" if args.resume else "run_manifest.json"
    manifest_path = os.path.join(args.exp_dir, manifest_name)
    runtime_manifest = _runtime_manifest(args, data_info)
    _json_dump(runtime_manifest, manifest_path)
    plot_client_data_distribution(
        data_info["client_splits"],
        data_info["train_coco"],
        os.path.join(args.exp_dir, "client_data_distribution.png"),
    )

    original_stdout, original_stderr = sys.stdout, sys.stderr
    tee_stdout = TeeOutput(args.log_file, original_stdout)
    tee_stderr = TeeOutput(args.log_file, original_stderr)
    sys.stdout, sys.stderr = tee_stdout, tee_stderr
    try:
        print(
            f"[Run] mode={args.mode}, method={args.fl_method}, seed={args.seed}, "
            f"partition_seed={args.partition_seed}, device={args.device}"
        )
        if args.mode == "solo":
            results = run_solo(args, data_info)
            _plot_standalone_history(results, args)
            save_results_json(results, os.path.join(args.exp_dir, "solo_results.json"))
        elif args.mode == "centralized":
            results = run_centralized(args, data_info)
            _plot_standalone_history(results, args)
            save_results_json(
                results, os.path.join(args.exp_dir, "centralized_results.json")
            )
        else:
            if args.resume:
                try:
                    from trainers.fl_server import evaluate_federated_checkpoint
                except ImportError as error:
                    raise RuntimeError(
                        "This build does not provide FL checkpoint evaluation"
                    ) from error
                results = evaluate_federated_checkpoint(args, data_info, args.exp_dir)
            else:
                results = run_federated_learning(args, data_info)
        runtime_manifest["resolved_model"] = results.get("architecture", {})
        runtime_manifest["result_file_schema_version"] = results.get("result_schema_version")
        runtime_manifest["status"] = "complete"
        _json_dump(runtime_manifest, manifest_path)
        print(f"[Done] Results written under {args.exp_dir}")
        return results
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        tee_stdout.close()
        tee_stderr.close()


if __name__ == "__main__":
    main()
