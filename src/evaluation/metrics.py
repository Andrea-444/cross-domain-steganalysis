from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Pairs differenced to measure domain generalization. Each gap is
# same-domain minus cross-domain, computed inside one seed and only then
# averaged. Differencing the means instead would discard the pairing and
# inflate the spread with seed-to-seed variation.
# Each pair differences two directions inside one seed, so one trained model
# supplies both terms. Two kinds answer two different questions.
#
# "transfer" holds a detector fixed and changes the test domain away from its
# own. The first direction tests it at home and the second tests it abroad. It
# measures how much the detector loses when the cover source changes.
#
# "test_domain" holds a detector fixed that trained on both domains, and
# changes the test domain between them. Neither domain is abroad for it, and
# neither has a training advantage, so the difference isolates the cover
# domain itself. The transfer pairs cannot do this: `real_to_real` against
# `ai_to_ai` changes the training domain and the test domain together.
GAP_PAIRS = {
    "real_to_real_minus_real_to_ai": {
        "first": "real_to_real", "second": "real_to_ai", "kind": "transfer",
    },
    "ai_to_ai_minus_ai_to_real": {
        "first": "ai_to_ai", "second": "ai_to_real", "kind": "transfer",
    },
    "mixed_to_real_minus_mixed_to_ai": {
        "first": "mixed_to_real", "second": "mixed_to_ai", "kind": "test_domain",
    },
}


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a rate, or None when the denominator is zero.

    A rate with no denominator is undefined rather than zero. Returning 0.0
    would read as a perfect score.
    """
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def equal_error_rate(
    labels: list[int], probabilities: list[float]
) -> float | None:
    """Return the rate where the false positive and false negative rates meet.

    Returns None when the labels hold a single class or no samples, because the
    ROC curve is undefined there.
    """
    if not labels or len(set(labels)) < 2:
        return None
    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    false_negative = 1.0 - true_positive
    difference = false_negative - false_positive
    crossing = np.flatnonzero(np.diff(np.signbit(difference)))
    if crossing.size == 0:
        index = int(np.nanargmin(np.abs(difference)))
        return float((false_positive[index] + false_negative[index]) / 2.0)
    # Interpolate between the two points that bracket the crossing, which is
    # more accurate than taking the nearer of them.
    first = int(crossing[0])
    second = first + 1
    span = difference[first] - difference[second]
    weight = 0.0 if span == 0 else difference[first] / span
    rate = false_positive[first] + weight * (false_positive[second] - false_positive[first])
    return float(rate)


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[float] | None = None,
) -> dict[str, object]:
    """Calculate binary clean-versus-stego classification metrics.

    When stego probabilities are supplied, threshold-independent metrics are
    added. Both are ``None`` when the labels contain a single class, which
    happens in small smoke runs and leaves ROC/PR undefined.
    """
    if not labels:
        # sklearn raises on an empty input. An empty evaluation has no value
        # for any metric, so every field is undefined rather than zero.
        empty: dict[str, object] = {
            name: None
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "specificity", "false_positive_rate", "false_negative_rate",
                "detection_error", "mcc",
            )
        }
        empty["confusion_matrix"] = [[0, 0], [0, 0]]
        if probabilities is not None:
            empty.update({"roc_auc": None, "pr_auc": None, "eer": None})
        return empty

    metrics: dict[str, object] = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }
    true_clean, false_stego, missed_stego, true_stego = (
        int(value)
        for value in np.asarray(metrics["confusion_matrix"], dtype=np.int64).ravel()
    )
    specificity = _ratio(true_clean, true_clean + false_stego)
    false_positive_rate = _ratio(false_stego, false_stego + true_clean)
    false_negative_rate = _ratio(missed_stego, missed_stego + true_stego)
    metrics["specificity"] = specificity
    metrics["false_positive_rate"] = false_positive_rate
    metrics["false_negative_rate"] = false_negative_rate
    metrics["detection_error"] = (
        None
        if false_positive_rate is None or false_negative_rate is None
        else 0.5 * (false_positive_rate + false_negative_rate)
    )
    metrics["mcc"] = (
        float(matthews_corrcoef(labels, predictions)) if labels else None
    )
    if probabilities is not None:
        single_class = len(set(labels)) < 2
        metrics["roc_auc"] = (
            None if single_class else float(roc_auc_score(labels, probabilities))
        )
        metrics["pr_auc"] = (
            None if single_class else float(average_precision_score(labels, probabilities))
        )
        metrics["eer"] = None if single_class else equal_error_rate(labels, probabilities)
    return metrics


def generalization_gaps(runs: list[dict[str, object]]) -> dict[str, object]:
    """Difference the two directions of each pair in ``GAP_PAIRS``.

    The difference is taken inside one seed and only then averaged, so the
    same trained model supplies both terms. The sign convention is always
    ``first - second``.

    For a metric where a higher value is better, a positive gap on a
    ``transfer`` pair means the detector degraded when the cover source
    changed. A positive gap on a ``test_domain`` pair means the second domain
    is harder for a detector that has no advantage in either. For an error
    metric such as ``detection_error`` or ``eer`` the sign reverses.
    """
    gaps: dict[str, object] = {}
    for name, pair in GAP_PAIRS.items():
        first, second = pair["first"], pair["second"]
        paired: dict[str, list[float]] = {}
        for run in runs:
            directions = run.get("directions", {})
            if first not in directions or second not in directions:
                continue
            for metric, value in directions[first].items():
                other = directions[second].get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if not isinstance(other, (int, float)) or isinstance(other, bool):
                    continue
                paired.setdefault(metric, []).append(float(value) - float(other))
        if not paired:
            continue
        entry: dict[str, object] = {
            "first_direction": first,
            "second_direction": second,
            "kind": pair["kind"],
        }
        for metric, values in paired.items():
            entry[f"{metric}_per_seed"] = values
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values))
        entry["seeds"] = len(next(iter(paired.values())))
        gaps[name] = entry
    return gaps


def summarize_directions(
    runs: list[dict[str, object]],
    directions: Iterable[str],
) -> dict[str, object]:
    """Aggregate repeated-seed metrics for each train-test direction."""
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    # Rates are undefined when their denominator is zero, so they aggregate
    # with the optional metrics rather than the required ones.
    # Threshold-independent metrics are optional: they are absent when no
    # probabilities were recorded, and None when a direction had one label class.
    optional_names = [
        "roc_auc", "pr_auc", "eer", "specificity", "false_positive_rate",
        "false_negative_rate", "detection_error", "mcc",
    ]
    summary: dict[str, object] = {}
    for direction in directions:
        direction_summary: dict[str, object] = {}
        for metric in metric_names:
            values = [
                float(run["directions"][direction][metric])
                for run in runs
                if run["directions"][direction].get(metric) is not None
            ]
            if not values:
                continue
            direction_summary[f"{metric}_mean"] = float(np.mean(values))
            direction_summary[f"{metric}_std"] = float(np.std(values))
        for metric in optional_names:
            values = [
                float(run["directions"][direction][metric])
                for run in runs
                if run["directions"][direction].get(metric) is not None
            ]
            if not values:
                continue
            direction_summary[f"{metric}_mean"] = float(np.mean(values))
            direction_summary[f"{metric}_std"] = float(np.std(values))
            direction_summary[f"{metric}_seeds"] = len(values)
        matrices = [
            np.asarray(run["directions"][direction]["confusion_matrix"], dtype=np.int64)
            for run in runs
            if run["directions"][direction].get("confusion_matrix") is not None
        ]
        if matrices:
            direction_summary["confusion_matrix_sum"] = np.sum(matrices, axis=0).tolist()
        summary[direction] = direction_summary
    gaps = generalization_gaps(runs)
    if gaps:
        summary["generalization_gaps"] = gaps
    return summary
