"""Command-line configuration for the AOD-4 RT-DETR federated study."""

from __future__ import annotations

import argparse
import math
import os


DEFAULT_DATA_ROOT = "/home/gpuadmin/kim/project2/data/aod4/AOD4/Images"


def _boolean_optional(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    """Add a paired --foo/--no-foo option."""
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="AOD-4 UAV detection with RT-DETR, LoRA and federated learning"
    )

    # Data and outputs
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_file", type=str, required=True,
                        help="JSON produced by scripts/prepare_split.py")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default="default")

    # Dataset / preprocessing. Small targets are retained in the primary study.
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--min_bbox_area", type=float, default=0.0,
                        help="Must match split metadata; 0 retains small objects")
    parser.add_argument("--min_bbox_side", type=float, default=0.0,
                        help="Must match split metadata; 0 retains small objects")
    parser.add_argument(
        "--rehash_source_images", action="store_true",
        help="Ignore the stat-validated cache and re-hash every source image before this run",
    )
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--partition", choices=["dirichlet", "iid"], default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.4)

    # Model / LoRA
    parser.add_argument("--model_name", choices=["rtdetr-l", "rtdetr-x"], default="rtdetr-l")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Local .pt path; defaults to <model_name>.pt")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Default 2*rank keeps alpha/r=2 across rank sensitivity runs")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    _boolean_optional(parser, "--apply_lora_backbone", True,
                      "Apply kernel-aware LoRA only to eligible CNN backbone convolutions")
    _boolean_optional(parser, "--apply_lora_decoder", True,
                      "Apply LoRA to decoder self-attention Q/K/V and cross-attention value projection")
    parser.add_argument("--backbone_min_channels", type=int, default=64)

    # Experiment modes
    parser.add_argument("--mode", choices=["solo", "centralized", "fl"], default="fl")
    parser.add_argument(
        "--fl_method",
        choices=["full_ft", "lora", "fedsa_lora", "fixed_share_b_lora"],
        default="fedsa_lora",
    )
    parser.add_argument("--client_id", type=int, default=None,
                        help="0-indexed client for solo mode")

    # Fixed primary budgets: 20 x 5 = 100 local epochs, matching solo/centralized.
    parser.add_argument("--fl_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--centralized_epochs", type=int, default=100)
    parser.add_argument("--solo_epochs", type=int, default=100)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: 1e-4 for full FT, 3e-4 for LoRA adapters")
    parser.add_argument("--head_lr", type=float, default=1e-4,
                        help="Learning rate for globally shared AOD-4 task heads")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1,
                        help="Full-FT backbone LR / head LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument(
        "--close_mosaic_epochs", type=int, default=10,
        help="Disable mosaic/mixup/cutmix/copy-paste for the final effective epochs",
    )
    parser.add_argument("--patience", type=int, default=0,
                        help="Validation checks without improvement; 0 disables early stopping (primary)")
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--fedprox_mu", type=float, default=0.0,
                        help="0 is FedAvg; nonzero is an explicit FedProx ablation")
    _boolean_optional(parser, "--reset_optimizer_each_round", True,
                      "Reset local AdamW state after each server broadcast")
    _boolean_optional(parser, "--amp", False,
                      "Use CUDA AMP (off in the primary RT-DETR protocol for stability)")

    # Evaluation, MIA and visualization
    parser.add_argument("--run_mia", action="store_true")
    parser.add_argument("--mia_max_samples", type=int, default=1000,
                        help="Maximum member and non-member images per client before attack splitting")
    parser.add_argument("--mia_calibration_fraction", type=float, default=0.5)
    parser.add_argument("--visualize_interval", type=int, default=5,
                        help="Save detection examples every N epochs/rounds; 0 disables")
    parser.add_argument("--vis_samples", type=int, default=6)
    _boolean_optional(parser, "--cross_client_eval", True,
                      "Evaluate each personalized/local model on every client test partition")

    # Runtime / reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition_seed", type=int, default=None,
                        help="Seed embedded in the split manifest; defaults to --seed")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Experiment directory for checkpoint evaluation (training is skipped)")

    args = parser.parse_args(argv)

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.lr is None:
        args.lr = 1e-4 if args.fl_method == "full_ft" else 3e-4
    if args.lora_alpha is None:
        args.lora_alpha = 2.0 * args.lora_rank
    if args.partition_seed is None:
        args.partition_seed = args.seed

    finite_fields = (
        "dirichlet_alpha", "min_bbox_area", "min_bbox_side", "lora_alpha",
        "lora_dropout", "lr", "head_lr", "backbone_lr_ratio", "weight_decay",
        "warmup_epochs", "min_lr_ratio", "grad_clip_norm", "fedprox_mu",
        "mia_calibration_fraction",
    )
    nonfinite = [name for name in finite_fields if not math.isfinite(float(getattr(args, name)))]
    if nonfinite:
        parser.error(f"Numeric arguments must be finite; invalid: {', '.join(nonfinite)}")
    if args.num_clients <= 0:
        parser.error("--num_clients must be positive")
    if args.num_clients < 2 and args.mode == "fl":
        parser.error("--num_clients must be >= 2 for federated learning")
    if args.seed < 0 or args.partition_seed < 0:
        parser.error("--seed and --partition_seed must be non-negative")
    if args.partition == "dirichlet" and args.dirichlet_alpha <= 0:
        parser.error("--dirichlet_alpha must be > 0")
    if args.lora_rank <= 0:
        parser.error("--lora_rank must be > 0")
    if args.lora_alpha <= 0:
        parser.error("--lora_alpha must be > 0")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora_dropout must be in [0, 1)")
    if args.apply_lora_decoder and args.lora_dropout != 0:
        parser.error("Fused RT-DETR Q/K/V LoRA requires --lora_dropout 0")
    if args.fl_method != "full_ft" and not (
        args.apply_lora_backbone or args.apply_lora_decoder
    ):
        parser.error("At least one LoRA target must be enabled")
    if args.fl_method in ("fedsa_lora", "fixed_share_b_lora") and args.mode != "fl":
        parser.error(
            f"--fl_method {args.fl_method} is defined only for --mode fl; "
            "use --fl_method lora for solo or centralized adapter training"
        )
    if args.mode == "solo" and args.client_id is None:
        parser.error("--client_id is required in solo mode")
    if args.client_id is not None and not 0 <= args.client_id < args.num_clients:
        parser.error("--client_id is outside the configured client range")
    if min(args.fl_rounds, args.local_epochs, args.centralized_epochs, args.solo_epochs) <= 0:
        parser.error("All epoch and round budgets must be positive")
    active_epoch_budget = (
        args.fl_rounds * args.local_epochs
        if args.mode == "fl"
        else args.solo_epochs if args.mode == "solo" else args.centralized_epochs
    )
    if args.close_mosaic_epochs > active_epoch_budget:
        parser.error(
            "--close_mosaic_epochs cannot exceed the active effective-epoch budget "
            f"({active_epoch_budget})"
        )
    if args.batch_size <= 0 or args.img_size <= 0 or args.num_workers < 0:
        parser.error("Invalid batch/image/worker configuration")
    if args.img_size % 32 != 0:
        parser.error("--img_size must be divisible by the RT-DETR maximum stride (32)")
    if args.model_weights and not os.path.isfile(args.model_weights):
        parser.error(f"--model_weights does not exist: {args.model_weights}")
    if args.num_classes <= 0 or args.backbone_min_channels <= 0:
        parser.error("--num_classes and --backbone_min_channels must be positive")
    if args.min_bbox_area < 0 or args.min_bbox_side < 0:
        parser.error("Bounding-box thresholds must be non-negative")
    if args.lr <= 0 or args.head_lr <= 0 or args.backbone_lr_ratio <= 0:
        parser.error("Learning rates and --backbone_lr_ratio must be positive")
    if args.weight_decay < 0 or args.warmup_epochs < 0:
        parser.error("--weight_decay and --warmup_epochs must be non-negative")
    if args.close_mosaic_epochs < 0:
        parser.error("--close_mosaic_epochs must be non-negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min_lr_ratio must be in [0, 1]")
    if args.grad_clip_norm <= 0:
        parser.error("--grad_clip_norm must be positive")
    if args.fedprox_mu < 0:
        parser.error("--fedprox_mu must be non-negative")
    if args.patience < 0 or args.val_interval <= 0 or args.visualize_interval < 0:
        parser.error("patience/validation/visualization intervals are invalid")
    if not 0.0 < args.mia_calibration_fraction < 1.0:
        parser.error("--mia_calibration_fraction must be between 0 and 1")
    if args.mia_max_samples < 4:
        parser.error("--mia_max_samples must be >= 4 for disjoint balanced MIA splits")
    if args.vis_samples < 0 or (args.visualize_interval > 0 and args.vis_samples == 0):
        parser.error("--vis_samples must be positive when visualization is enabled")

    # A flat experiment directory matches the run scripts and result aggregator.
    args.exp_dir = os.path.abspath(os.path.join(args.output_dir, args.exp_name))
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_file_dir = os.path.abspath(args.log_dir)
    os.makedirs(args.log_file_dir, exist_ok=True)
    args.log_file = os.path.join(args.log_file_dir, f"{args.exp_name}.log")
    return args
