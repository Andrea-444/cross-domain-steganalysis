from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    if probabilities is not None:
        single_class = len(set(labels)) < 2
        metrics["roc_auc"] = (
            None if single_class else float(roc_auc_score(labels, probabilities))
        )
        metrics["pr_auc"] = (
            None if single_class else float(average_precision_score(labels, probabilities))
        )
    return metrics


def summarize_directions(
    runs: list[dict[str, object]],
    directions: Iterable[str],
) -> dict[str, object]:
    """Aggregate repeated-seed metrics for each train-test direction."""
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    # Threshold-independent metrics are optional: they are absent when no
    # probabilities were recorded, and None when a direction had one label class.
    optional_names = ["roc_auc", "pr_auc"]
    summary: dict[str, object] = {}
    for direction in directions:
        direction_summary: dict[str, object] = {}
        for metric in metric_names:
            values = [float(run["directions"][direction][metric]) for run in runs]
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
        ]
        direction_summary["confusion_matrix_sum"] = np.sum(matrices, axis=0).tolist()
        summary[direction] = direction_summary
    return summary
