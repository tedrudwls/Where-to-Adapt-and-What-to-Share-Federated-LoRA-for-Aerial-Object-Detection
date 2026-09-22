"""Balanced, calibration-separated image-level loss membership inference."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
CLIENT_ATTACK_SEED_STRIDE = 10_007


def client_attack_seed(training_seed: int, client_id: int) -> int:
    """Paired attack split seed shared by Solo, Centralized and FL methods."""
    if int(client_id) < 0:
        raise ValueError("client_id must be non-negative")
    return int(training_seed) + CLIENT_ATTACK_SEED_STRIDE * int(client_id)


def _validated_losses(values: List[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{label} losses are empty")
    if not np.all(np.isfinite(array)):
        bad = np.flatnonzero(~np.isfinite(array))[:10].tolist()
        raise FloatingPointError(f"{label} losses contain non-finite values at indices {bad}")
    return array


def _balanced_calibration_evaluation_split(
    member_losses: np.ndarray,
    nonmember_losses: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Balance classes first, then split each class into disjoint attack sets."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    balanced_size = min(member_losses.size, nonmember_losses.size)
    if balanced_size < 4:
        raise ValueError(
            "Balanced loss-MIA needs at least four member and four non-member images "
            f"(received {member_losses.size} and {nonmember_losses.size})"
        )

    rng = np.random.default_rng(int(seed))
    members = member_losses[rng.permutation(member_losses.size)[:balanced_size]]
    nonmembers = nonmember_losses[rng.permutation(nonmember_losses.size)[:balanced_size]]
    calibration_size = int(round(balanced_size * float(calibration_fraction)))
    calibration_size = min(max(calibration_size, 1), balanced_size - 1)
    return (
        members[:calibration_size],
        nonmembers[:calibration_size],
        members[calibration_size:],
        nonmembers[calibration_size:],
        balanced_size,
    )


def _labels(member_count: int, nonmember_count: int) -> np.ndarray:
    return np.concatenate((
        np.ones(member_count, dtype=np.int64),
        np.zeros(nonmember_count, dtype=np.int64),
    ))


def _raw_loss_scores(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> np.ndarray:
    """Return the conventional score before attacker direction calibration."""
    return np.concatenate((-member_losses, -nonmember_losses))


def _calibrate_direction(labels: np.ndarray, raw_scores: np.ndarray) -> Tuple[float, float]:
    """Select score sign using calibration AUC only; never inspect evaluation labels."""
    raw_auc = float(roc_auc_score(labels, raw_scores))
    multiplier = -1.0 if raw_auc < 0.5 else 1.0
    return multiplier, raw_auc


def _calibrate_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Choose the balanced-accuracy/Youden-J threshold on calibration data only."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, scores, drop_intermediate=False
    )
    objective = true_positive_rate - false_positive_rate
    best_value = float(np.max(objective))
    candidates = np.flatnonzero(np.isclose(objective, best_value, rtol=0.0, atol=1e-12))
    finite_candidates = [index for index in candidates if np.isfinite(thresholds[index])]
    chosen = finite_candidates[0] if finite_candidates else int(candidates[0])
    threshold = float(thresholds[chosen])
    predictions = (scores >= threshold).astype(np.int64)
    calibration_asr = float(np.mean(predictions == labels))
    return threshold, calibration_asr


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, maximum_fpr: float) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels, scores, drop_intermediate=False
    )
    admissible = true_positive_rate[false_positive_rate <= maximum_fpr + 1e-12]
    return float(np.max(admissible)) if admissible.size else 0.0


def _evaluation_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "tpr_at_1fpr": _tpr_at_fpr(labels, scores, maximum_fpr=0.01),
        "asr": float(np.mean(predictions == labels)),
    }


def _resample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, values.size, size=values.size)]


def _percentile_interval(values: List[float], confidence_level: float) -> dict:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return {"lower": float(lower), "upper": float(upper)}


def _bootstrap_attack_pipeline(
    calibration_members: np.ndarray,
    calibration_nonmembers: np.ndarray,
    evaluation_members: np.ndarray,
    evaluation_nonmembers: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    """Stratified bootstrap of calibration direction, threshold and evaluation."""
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(int(seed))
    values = {"auc_roc": [], "tpr_at_1fpr": [], "asr": []}
    calibration_labels = _labels(
        calibration_members.size, calibration_nonmembers.size
    )
    evaluation_labels = _labels(evaluation_members.size, evaluation_nonmembers.size)

    for _ in range(int(samples)):
        bootstrap_calibration_members = _resample(calibration_members, rng)
        bootstrap_calibration_nonmembers = _resample(calibration_nonmembers, rng)
        calibration_raw_scores = _raw_loss_scores(
            bootstrap_calibration_members, bootstrap_calibration_nonmembers
        )
        direction, _ = _calibrate_direction(calibration_labels, calibration_raw_scores)
        threshold, _ = _calibrate_threshold(
            calibration_labels, direction * calibration_raw_scores
        )

        bootstrap_evaluation_members = _resample(evaluation_members, rng)
        bootstrap_evaluation_nonmembers = _resample(evaluation_nonmembers, rng)
        evaluation_scores = direction * _raw_loss_scores(
            bootstrap_evaluation_members, bootstrap_evaluation_nonmembers
        )
        bootstrap_metrics = _evaluation_metrics(
            evaluation_labels, evaluation_scores, threshold
        )
        for key in values:
            values[key].append(bootstrap_metrics[key])

    return {
        "method": "stratified_nonparametric_full_attack_pipeline_percentile",
        "resamples": int(samples),
        "confidence_level": CONFIDENCE_LEVEL,
        "unit_of_resampling": "image_loss_within_each_membership_class",
        "recalibrates_direction_and_threshold": True,
        "conditional_on_balanced_subsample": True,
        **{
            key: _percentile_interval(metric_values, CONFIDENCE_LEVEL)
            for key, metric_values in values.items()
        },
    }


def compute_mia_metrics(
    member_losses: List[float],
    non_member_losses: List[float],
    calibration_fraction: float = 0.5,
    seed: int = 42,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> Dict[str, object]:
    """Evaluate a calibrated image-level detection-loss membership attack.

    Member and non-member examples are subsampled to an exactly balanced set.
    Each class is split into disjoint attack-calibration and attack-evaluation
    subsets. The calibration subset alone chooses whether low or high loss is
    more member-like and selects the threshold. AUC, TPR@1%FPR and threshold
    attack success rate are then measured on the untouched evaluation subset.
    """
    raw_members = _validated_losses(member_losses, "member")
    raw_nonmembers = _validated_losses(non_member_losses, "non-member")
    (
        calibration_members,
        calibration_nonmembers,
        evaluation_members,
        evaluation_nonmembers,
        balanced_size,
    ) = _balanced_calibration_evaluation_split(
        raw_members,
        raw_nonmembers,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )

    calibration_labels = _labels(
        calibration_members.size, calibration_nonmembers.size
    )
    calibration_raw_scores = _raw_loss_scores(
        calibration_members, calibration_nonmembers
    )
    direction_multiplier, calibration_raw_auc = _calibrate_direction(
        calibration_labels, calibration_raw_scores
    )
    calibration_scores = direction_multiplier * calibration_raw_scores
    threshold, calibration_asr = _calibrate_threshold(
        calibration_labels, calibration_scores
    )

    evaluation_labels = _labels(evaluation_members.size, evaluation_nonmembers.size)
    evaluation_scores = direction_multiplier * _raw_loss_scores(
        evaluation_members, evaluation_nonmembers
    )
    evaluation = _evaluation_metrics(evaluation_labels, evaluation_scores, threshold)
    bootstrap = _bootstrap_attack_pipeline(
        calibration_members,
        calibration_nonmembers,
        evaluation_members,
        evaluation_nonmembers,
        samples=int(bootstrap_samples),
        seed=int(seed) + 1_000_003,
    )

    conventional_direction_selected = direction_multiplier > 0
    score_name = "negative_detection_loss" if conventional_direction_selected else "detection_loss"
    score_direction = (
        "lower_loss_is_more_likely_member"
        if conventional_direction_selected
        else "higher_loss_is_more_likely_member"
    )
    auc_roc = evaluation["auc_roc"]
    return {
        "attack": "image_level_ground_truth_matched_detection_loss_threshold",
        "score": score_name,
        "score_direction": score_direction,
        "direction_selected_on": "calibration_auc_only",
        "calibration_raw_negative_loss_auc": float(calibration_raw_auc),
        "calibration_oriented_auc": float(
            roc_auc_score(calibration_labels, calibration_scores)
        ),
        "balanced": True,
        "calibration_evaluation_disjoint": True,
        "seed": int(seed),
        "requested_calibration_fraction": float(calibration_fraction),
        "actual_calibration_fraction": float(calibration_members.size / balanced_size),
        "raw_member_count": int(raw_members.size),
        "raw_nonmember_count": int(raw_nonmembers.size),
        "balanced_count_per_class": int(balanced_size),
        "calibration_member_count": int(calibration_members.size),
        "calibration_nonmember_count": int(calibration_nonmembers.size),
        "evaluation_member_count": int(evaluation_members.size),
        "evaluation_nonmember_count": int(evaluation_nonmembers.size),
        "evaluation_fpr_resolution": float(1.0 / evaluation_nonmembers.size),
        "tpr_at_1fpr_empirically_resolved": bool(evaluation_nonmembers.size >= 100),
        "tpr_at_1fpr_resolution_note": (
            "Fewer than 100 evaluation non-members: empirical FPR cannot resolve a "
            "nonzero 1% operating point; the statistic is effectively TPR at 0% FPR."
            if evaluation_nonmembers.size < 100
            else "Empirical evaluation has at least 100 non-members."
        ),
        "threshold_in_oriented_score_units": float(threshold),
        "threshold": float(threshold),
        "calibration_asr": float(calibration_asr),
        "auc_roc": float(auc_roc),
        "tpr_at_1fpr": float(evaluation["tpr_at_1fpr"]),
        "asr": float(evaluation["asr"]),
        "auc_advantage_over_random": float(auc_roc - 0.5),
        "auc_gini_advantage": float(2.0 * auc_roc - 1.0),
        "bootstrap_95_ci": bootstrap,
        "evaluation_member_loss_mean": float(evaluation_members.mean()),
        "evaluation_nonmember_loss_mean": float(evaluation_nonmembers.mean()),
        "random_baseline_auc": 0.5,
        "random_baseline_tpr_at_1fpr": 0.01,
        "random_baseline_asr": 0.5,
        "interpretation_note": (
            "This is an empirical white-box leakage audit, not a formal privacy guarantee; "
            "train/test distribution shift can confound a loss-threshold attack."
        ),
    }
