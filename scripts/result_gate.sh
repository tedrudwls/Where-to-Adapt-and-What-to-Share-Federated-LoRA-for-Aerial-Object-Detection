#!/usr/bin/env bash
# Shared result-reuse gate. Source this file after PROJECT_DIR and PYTHON_BIN are set.

result_matches_cli() {
    local result_file="$1"
    shift
    local require_mia=0
    if [[ "${1:-}" == "--require-mia" ]]; then
        require_mia=1
        shift
    fi
    local -a launch_args=("$@")

    # Defaults mirror configs/config.py so this helper also works for concise
    # legacy launch commands. Every publication suite passes these explicitly.
    local mode="fl"
    local method="fedsa_lora"
    local seed="42"
    local partition_seed=""
    local partition="dirichlet"
    local dirichlet_alpha="0.4"
    local model_name="rtdetr-l"
    local model_weights=""
    local num_classes="4"
    local batch_size="8"
    local img_size="640"
    local num_workers="4"
    local num_clients="3"
    local lora_rank="8"
    local lora_alpha=""
    local lora_dropout="0.0"
    local apply_lora_backbone="true"
    local apply_lora_decoder="true"
    local backbone_min_channels="64"
    local fl_rounds="20"
    local local_epochs="5"
    local centralized_epochs="100"
    local solo_epochs="100"
    local client_id=""
    local split_file=""
    local learning_rate=""
    local head_lr="0.0001"
    local backbone_lr_ratio="0.1"
    local weight_decay="0.0001"
    local warmup_epochs="5.0"
    local min_lr_ratio="0.01"
    local grad_clip_norm="0.1"
    local close_mosaic_epochs="10"
    local fedprox_mu="0.0"
    local reset_optimizer_each_round="true"
    local amp="false"
    local patience="0"
    local val_interval="5"
    local cross_client_eval="true"
    local visualize_interval="5"
    local vis_samples="6"
    local mia_max_samples="1000"
    local mia_calibration_fraction="0.5"

    local index=0
    local option value
    while (( index < ${#launch_args[@]} )); do
        option="${launch_args[index]}"
        case "$option" in
            --apply_lora_backbone) apply_lora_backbone="true" ;;
            --no-apply_lora_backbone) apply_lora_backbone="false" ;;
            --apply_lora_decoder) apply_lora_decoder="true" ;;
            --no-apply_lora_decoder) apply_lora_decoder="false" ;;
            --reset_optimizer_each_round) reset_optimizer_each_round="true" ;;
            --no-reset_optimizer_each_round) reset_optimizer_each_round="false" ;;
            --amp) amp="true" ;;
            --no-amp) amp="false" ;;
            --cross_client_eval) cross_client_eval="true" ;;
            --no-cross_client_eval) cross_client_eval="false" ;;
            --mode|--fl_method|--seed|--partition_seed|--partition|--dirichlet_alpha|\
            --model_name|--model_weights|--num_classes|--batch_size|--img_size|--num_workers|--num_clients|\
            --lora_rank|--lora_alpha|--lora_dropout|--backbone_min_channels|\
            --fl_rounds|--local_epochs|--centralized_epochs|--solo_epochs|--client_id|\
            --split_file|--lr|--head_lr|--backbone_lr_ratio|--weight_decay|--warmup_epochs|\
            --min_lr_ratio|--grad_clip_norm|--close_mosaic_epochs|--fedprox_mu|--patience|\
            --val_interval|--visualize_interval|--vis_samples|--mia_max_samples|--mia_calibration_fraction)
                if (( index + 1 >= ${#launch_args[@]} )); then
                    echo "[ERROR] Result gate: $option has no value" >&2
                    return 2
                fi
                value="${launch_args[index + 1]}"
                case "$option" in
                    --mode) mode="$value" ;;
                    --fl_method) method="$value" ;;
                    --seed) seed="$value" ;;
                    --partition_seed) partition_seed="$value" ;;
                    --partition) partition="$value" ;;
                    --dirichlet_alpha) dirichlet_alpha="$value" ;;
                    --model_name) model_name="$value" ;;
                    --model_weights) model_weights="$value" ;;
                    --num_classes) num_classes="$value" ;;
                    --batch_size) batch_size="$value" ;;
                    --img_size) img_size="$value" ;;
                    --num_workers) num_workers="$value" ;;
                    --num_clients) num_clients="$value" ;;
                    --lora_rank) lora_rank="$value" ;;
                    --lora_alpha) lora_alpha="$value" ;;
                    --lora_dropout) lora_dropout="$value" ;;
                    --backbone_min_channels) backbone_min_channels="$value" ;;
                    --fl_rounds) fl_rounds="$value" ;;
                    --local_epochs) local_epochs="$value" ;;
                    --centralized_epochs) centralized_epochs="$value" ;;
                    --solo_epochs) solo_epochs="$value" ;;
                    --client_id) client_id="$value" ;;
                    --split_file) split_file="$value" ;;
                    --lr) learning_rate="$value" ;;
                    --head_lr) head_lr="$value" ;;
                    --backbone_lr_ratio) backbone_lr_ratio="$value" ;;
                    --weight_decay) weight_decay="$value" ;;
                    --warmup_epochs) warmup_epochs="$value" ;;
                    --min_lr_ratio) min_lr_ratio="$value" ;;
                    --grad_clip_norm) grad_clip_norm="$value" ;;
                    --close_mosaic_epochs) close_mosaic_epochs="$value" ;;
                    --fedprox_mu) fedprox_mu="$value" ;;
                    --patience) patience="$value" ;;
                    --val_interval) val_interval="$value" ;;
                    --visualize_interval) visualize_interval="$value" ;;
                    --vis_samples) vis_samples="$value" ;;
                    --mia_max_samples) mia_max_samples="$value" ;;
                    --mia_calibration_fraction) mia_calibration_fraction="$value" ;;
                esac
                index=$((index + 1))
                ;;
        esac
        index=$((index + 1))
    done

    if [[ -z "$partition_seed" ]]; then
        partition_seed="$seed"
    fi
    if [[ -z "$lora_alpha" ]]; then
        lora_alpha=$((2 * lora_rank))
    fi
    if [[ -z "$learning_rate" ]]; then
        if [[ "$method" == "full_ft" ]]; then
            learning_rate="0.0001"
        else
            learning_rate="0.0003"
        fi
    fi
    if [[ -z "$split_file" ]]; then
        echo "[ERROR] Result gate: launch command has no --split_file" >&2
        return 2
    fi

    local checker_path="${PROJECT_DIR}/scripts/result_complete.py"
    local -a checker=(
        "$PYTHON_BIN" "$checker_path" "$result_file"
        --expect_split_file "$split_file"
        --expect "mode=$mode"
        --expect "fl_method=$method"
        --expect "seed=$seed"
        --expect "partition_seed=$partition_seed"
        --expect "partition=$partition"
        --expect "num_clients=$num_clients"
        --expect "training_experiment.fl_method=$method"
        --expect "training_experiment.model_name=$model_name"
        --expect "training_experiment.num_clients=$num_clients"
        --expect "training_experiment.partition=$partition"
        --expect "training_experiment.seed=$seed"
        --expect "training_experiment.partition_seed=$partition_seed"
        --expect "training_experiment.batch_size=$batch_size"
        --expect "training_experiment.img_size=$img_size"
        --expect "training_experiment.num_workers=$num_workers"
        --expect "training_experiment.lr=$learning_rate"
        --expect "training_experiment.head_lr=$head_lr"
        --expect "training_experiment.backbone_lr_ratio=$backbone_lr_ratio"
        --expect "training_experiment.weight_decay=$weight_decay"
        --expect "training_experiment.warmup_epochs=$warmup_epochs"
        --expect "training_experiment.min_lr_ratio=$min_lr_ratio"
        --expect "training_experiment.grad_clip_norm=$grad_clip_norm"
        --expect "training_experiment.close_mosaic_epochs=$close_mosaic_epochs"
        --expect "training_experiment.fedprox_mu=$fedprox_mu"
        --expect "training_experiment.reset_optimizer_each_round=$reset_optimizer_each_round"
        --expect "training_experiment.amp=$amp"
        --expect "training_experiment.patience=$patience"
        --expect "training_experiment.val_interval=$val_interval"
        --expect "training_experiment.cross_client_eval=$cross_client_eval"
        --expect "training_experiment.visualize_interval=$visualize_interval"
        --expect "training_experiment.vis_samples=$vis_samples"
        --expect "training_experiment.optimizer=AdamW"
        --expect "training_experiment.lr_schedule=global_step_linear_warmup_then_cosine_decay"
        --expect "training_experiment.effective_close_mosaic_epochs=$close_mosaic_epochs"
        --expect "training_experiment.augmentation_protocol.implementation=ultralytics_8.4.126_RTDETRDataset"
        --expect "training_experiment.augmentation_protocol.initial.hsv_h=0.015"
        --expect "training_experiment.augmentation_protocol.initial.hsv_s=0.7"
        --expect "training_experiment.augmentation_protocol.initial.hsv_v=0.4"
        --expect "training_experiment.augmentation_protocol.initial.degrees=0.0"
        --expect "training_experiment.augmentation_protocol.initial.translate=0.1"
        --expect "training_experiment.augmentation_protocol.initial.scale=0.5"
        --expect "training_experiment.augmentation_protocol.initial.shear=0.0"
        --expect "training_experiment.augmentation_protocol.initial.perspective=0.0"
        --expect "training_experiment.augmentation_protocol.initial.flipud=0.0"
        --expect "training_experiment.augmentation_protocol.initial.fliplr=0.5"
        --expect "training_experiment.augmentation_protocol.initial.bgr=0.0"
        --expect "training_experiment.augmentation_protocol.initial.mosaic=1.0"
        --expect "training_experiment.augmentation_protocol.initial.mixup=0.0"
        --expect "training_experiment.augmentation_protocol.initial.cutmix=0.0"
        --expect "training_experiment.augmentation_protocol.initial.copy_paste=0.0"
        --expect "training_experiment.augmentation_protocol.initial.copy_paste_mode=flip"
        --expect "training_experiment.augmentation_protocol.close_mosaic_effective_epochs_requested=$close_mosaic_epochs"
        --expect 'training_experiment.augmentation_protocol.close_mosaic_disables=["mosaic","mixup","cutmix","copy_paste"]'
        --expect "training_experiment.augmentation_protocol.rect=false"
        --expect "training_experiment.augmentation_protocol.cache=false"
        --expect "training_experiment.augmentation_protocol.train_only=true"
        --expect "training_experiment.augmentation_protocol.persistent_dataloader_workers=false"
        --expect "architecture.ultralytics_version=8.4.126"
        --expect "architecture.model_name=$model_name"
        --expect "architecture.num_classes=$num_classes"
        --expect "architecture.fine_tuning_mode=$method"
    )
    if [[ -n "$model_weights" ]]; then
        checker+=(
            --expect "training_experiment.model_weights=$model_weights"
            --expect_file_sha256 architecture.model_weight_sha256 "$model_weights"
        )
    else
        checker+=(--expect "training_experiment.model_weights=null")
        local named_weight="${PROJECT_DIR}/${model_name}.pt"
        if [[ ! -f "$named_weight" ]]; then
            echo "[INVALID] Cannot verify pretrained weight provenance: $named_weight is missing" >&2
            echo "          Restore that exact file or set MODEL_WEIGHTS to an immutable absolute path." >&2
            return 1
        fi
        checker+=(
            --expect_file_sha256 architecture.model_weight_sha256 "$named_weight"
        )
    fi

    if [[ "$partition" == "dirichlet" ]]; then
        checker+=(
            --expect "dirichlet_alpha=$dirichlet_alpha"
            --expect "training_experiment.dirichlet_alpha=$dirichlet_alpha"
        )
    else
        checker+=(
            --expect "dirichlet_alpha=null"
            --expect "training_experiment.dirichlet_alpha=null"
        )
    fi
    if [[ "$method" != "full_ft" ]]; then
        checker+=(
            --expect "lora_rank=$lora_rank"
            --expect "lora_alpha=$lora_alpha"
            --expect "apply_lora_backbone=$apply_lora_backbone"
            --expect "apply_lora_decoder=$apply_lora_decoder"
            --expect "training_experiment.lora_rank=$lora_rank"
            --expect "training_experiment.lora_alpha=$lora_alpha"
            --expect "training_experiment.lora_dropout=$lora_dropout"
            --expect "training_experiment.apply_lora_backbone=$apply_lora_backbone"
            --expect "training_experiment.apply_lora_decoder=$apply_lora_decoder"
            --expect "training_experiment.backbone_min_channels=$backbone_min_channels"
        )
    fi
    case "$mode" in
        fl)
            checker+=(
                --expect "rounds_executed=$fl_rounds"
                --expect "local_epochs=$local_epochs"
                --expect "training_experiment.fl_rounds=$fl_rounds"
                --expect "training_experiment.local_epochs=$local_epochs"
                --expect "training_experiment.client_participation=all_clients_every_round"
                --expect "training_experiment.aggregation_weighting=local_train_image_count"
                --expect "training_experiment.nonfloating_state_policy=retain_previous_server_value"
                --expect "training_experiment.validation_frequency_rounds=1"
                --expect "training_experiment.selection_criterion=macro_client_local_validation_AP"
                --expect "training_experiment.local_epoch_budget_per_client=$((fl_rounds * local_epochs))"
            )
            ;;
        centralized)
            checker+=(
                --expect "training_experiment.mode=centralized"
                --expect "training_experiment.centralized_epochs=$centralized_epochs"
                --expect "training_experiment.checkpoint_selection=macro_client_local_validation_AP"
            )
            ;;
        solo)
            checker+=(
                --expect "training_experiment.mode=solo"
                --expect "training_experiment.solo_epochs=$solo_epochs"
                --expect "training_experiment.checkpoint_selection=single_client_validation_AP"
            )
            if [[ -z "$client_id" ]]; then
                echo "[ERROR] Result gate: solo launch command has no --client_id" >&2
                return 2
            fi
            checker+=(--expect "client_id=$client_id")
            ;;
        *)
            echo "[ERROR] Result gate: unsupported mode $mode" >&2
            return 2
            ;;
    esac
    if [[ "$require_mia" == "1" ]]; then
        checker+=(
            --require_mia
            --expect_mia_max_samples "$mia_max_samples"
            --expect_mia_calibration_fraction "$mia_calibration_fraction"
        )
    fi
    "${checker[@]}"
}
