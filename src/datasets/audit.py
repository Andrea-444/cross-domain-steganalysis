"""Audit whether two cover domains are separable before any embedding.

The research claim depends on natural and AI covers differing *only* in how
they were generated. If they also differ in compression history or resampling
ratio, a steganalyzer can learn that instead, and every cross-domain number
becomes uninterpretable.

This module answers one question: given only clean covers and no stego at all,
how well can a classifier tell the domains apart? A score near 0.5 means the
normalization is doing its job. A score near 1.0 means the pair is confounded
and no cross-domain result from it can be trusted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.manifest import read_manifest


DOMAIN_LABELS = {"real": 0, "ai": 1}


def blockiness_score(array: np.ndarray) -> float:
    """Bounded contrast between discontinuity at 8-pixel boundaries and within.

    JPEG quantizes each 8x8 block independently, leaving steps at the block
    edges. A lossless image has no reason to prefer those columns.

    Returns a value in [-1, 1]: 0 means boundaries look like everywhere else,
    and positive values mean the discontinuity concentrates at block edges.
    A ratio would be unbounded and undefined for a flat block interior, which
    is precisely the most blocky case, so the normalized contrast is used.
    """
    array = array.astype(np.float64)
    horizontal = np.abs(np.diff(array, axis=1))
    vertical = np.abs(np.diff(array, axis=0))
    # diff index j holds |col[j+1] - col[j]|, so a boundary before column 8k
    # sits at index 8k-1.
    h_cols = np.arange(horizontal.shape[1])
    v_rows = np.arange(vertical.shape[0])
    h_edge = (h_cols + 1) % 8 == 0
    v_edge = (v_rows + 1) % 8 == 0
    edge = np.concatenate([horizontal[:, h_edge].ravel(), vertical[v_edge, :].ravel()])
    interior = np.concatenate(
        [horizontal[:, ~h_edge].ravel(), vertical[~v_edge, :].ravel()]
    )
    edge_mean = float(edge.mean())
    interior_mean = float(interior.mean())
    total = edge_mean + interior_mean
    if total <= 1e-9:
        return 0.0  # a uniform image has no discontinuity anywhere
    return float((edge_mean - interior_mean) / total)


def highpass_energy(array: np.ndarray) -> float:
    """Standard deviation of a Laplacian residual.

    Steganalysis lives in this band, and it is exactly what lossy compression
    and downsampling attenuate.
    """
    array = array.astype(np.float64)
    laplacian = (
        4.0 * array[1:-1, 1:-1]
        - array[:-2, 1:-1]
        - array[2:, 1:-1]
        - array[1:-1, :-2]
        - array[1:-1, 2:]
    )
    return float(laplacian.std())


def neighbour_residual_std(array: np.ndarray) -> float:
    """Spread of the difference from a local 4-neighbour mean."""
    array = array.astype(np.float64)
    mean = (
        array[:-2, 1:-1] + array[2:, 1:-1] + array[1:-1, :-2] + array[1:-1, 2:]
    ) / 4.0
    return float((array[1:-1, 1:-1] - mean).std())


def image_features(array: np.ndarray) -> dict[str, float]:
    return {
        "blockiness": blockiness_score(array),
        "highpass_energy": highpass_energy(array),
        "residual_std": neighbour_residual_std(array),
    }


def load_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.float64)


def audit_domain_separability(
    manifest: str | Path,
    folds: int = 5,
    seed: int = 42,
    path_field: str = "clean_path",
) -> dict[str, object]:
    """Classify domain from clean covers alone and report how easy it was."""
    records = read_manifest(manifest)
    rows = [r for r in records if str(r.get("domain")) in DOMAIN_LABELS]
    if not rows:
        raise ValueError("Manifest contains no real/ai domain records.")

    feature_names = ["blockiness", "highpass_energy", "residual_std"]
    matrix: list[list[float]] = []
    labels: list[int] = []
    per_domain: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: dict[str, int] = defaultdict(int)

    for record in rows:
        domain = str(record["domain"])
        features = image_features(load_grayscale(str(record[path_field])))
        matrix.append([features[name] for name in feature_names])
        labels.append(DOMAIN_LABELS[domain])
        counts[domain] += 1
        for name, value in features.items():
            per_domain[domain][name].append(value)

    features_array = np.asarray(matrix, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    # Cross-validated so the score reflects generalization, not memorization.
    smallest_class = int(min(np.bincount(labels_array)))
    folds = max(2, min(folds, smallest_class))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    predictions = np.zeros_like(labels_array)
    scores = np.zeros(len(labels_array), dtype=np.float64)
    for train_index, test_index in splitter.split(features_array, labels_array):
        model.fit(features_array[train_index], labels_array[train_index])
        predictions[test_index] = model.predict(features_array[test_index])
        scores[test_index] = model.predict_proba(features_array[test_index])[:, 1]

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    return {
        "manifest": str(Path(manifest).resolve()),
        "counts": dict(counts),
        "folds": folds,
        "seed": seed,
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "roc_auc": float(roc_auc_score(labels_array, scores)),
        "features": feature_names,
        "per_domain": {
            domain: {name: float(np.mean(values)) for name, values in stats.items()}
            for domain, stats in per_domain.items()
        },
        "per_domain_std": {
            domain: {name: float(np.std(values)) for name, values in stats.items()}
            for domain, stats in per_domain.items()
        },
    }


def add_arguments(subparsers) -> None:
    audit = subparsers.add_parser(
        "audit-domains",
        help="Test whether clean covers alone reveal their domain.",
    )
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--path-field", default="clean_path")
    audit.add_argument("--folds", type=int, default=5)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--output", default=None)


def run(args: argparse.Namespace) -> dict[str, object]:
    result = audit_domain_separability(
        manifest=args.manifest,
        folds=args.folds,
        seed=args.seed,
        path_field=args.path_field,
    )
    print("\nDomain separability on clean covers only")
    print(f"  counts            : {result['counts']}")
    print(f"  balanced accuracy : {result['balanced_accuracy']:.3f}")
    print(f"  ROC-AUC           : {result['roc_auc']:.3f}")
    print("\n  feature means by domain")
    for name in result["features"]:
        real = result["per_domain"].get("real", {}).get(name, float("nan"))
        ai = result["per_domain"].get("ai", {}).get(name, float("nan"))
        print(f"    {name:<16} real={real:10.4f}   ai={ai:10.4f}")
    print(
        "\n  0.5 means the domains are indistinguishable before embedding."
        "\n  Near 1.0 means a detector can identify the domain from artifacts alone."
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Written: {path.resolve()}")
    return result
