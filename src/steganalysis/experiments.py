from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.datasets.manifest import read_manifest, validate_stego_manifest
from src.steganalysis.srnet import (
    DEFAULT_SRNET_ROOT,
    SRNET_REFERENCE_COMMIT,
    predict_srnet,
    train_srnet,
)


DIRECTIONS = {
    "ai_to_ai": ("ai", "ai"),
    "real_to_real": ("real", "real"),
    "ai_to_real": ("ai", "real"),
    "real_to_ai": ("real", "ai"),
}


def calculate_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary: dict[str, object] = {}
    for direction in DIRECTIONS:
        direction_summary: dict[str, object] = {}
        for metric in metric_names:
            values = [float(run["directions"][direction][metric]) for run in runs]
            direction_summary[f"{metric}_mean"] = float(np.mean(values))
            direction_summary[f"{metric}_std"] = float(np.std(values))
        matrices = [
            np.asarray(run["directions"][direction]["confusion_matrix"], dtype=np.int64)
            for run in runs
        ]
        direction_summary["confusion_matrix_sum"] = np.sum(matrices, axis=0).tolist()
        summary[direction] = direction_summary
    return summary


def validate_pair_splits(records: list[dict[str, object]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        image_id = str(record["image_id"])
        split = str(record["split"])
        previous = seen.setdefault(image_id, split)
        if previous != split:
            raise ValueError(f"Pair leakage: {image_id} appears in {previous} and {split}.")


def timestamped_results_dir(output: str | Path, detector: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output) / f"{detector}_{timestamp}"
    suffix = 1
    while path.exists():
        path = Path(output) / f"{detector}_{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def run_srnet_protocol(
    manifest: str | Path,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output: str | Path,
    reference_root: str | Path,
    device: str | None,
) -> Path:
    records = read_manifest(manifest)
    validate_stego_manifest(records)
    validate_pair_splits(records)
    run_dir = timestamped_results_dir(output, "srnet")
    runs: list[dict[str, object]] = []

    for seed in seeds:
        models: dict[str, tuple[object, torch.device]] = {}
        for train_domain in ("real", "ai"):
            models[train_domain] = train_srnet(
                records=records,
                train_domain=train_domain,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                reference_root=reference_root,
                device_name=device,
            )
        direction_results: dict[str, object] = {}
        for name, (train_domain, test_domain) in DIRECTIONS.items():
            model, model_device = models[train_domain]
            labels, predictions, _ = predict_srnet(
                model=model,
                records=records,
                test_domain=test_domain,
                batch_size=batch_size,
                device=model_device,
            )
            direction_results[name] = calculate_metrics(labels, predictions)
        runs.append({"seed": seed, "directions": direction_results})

    result = {
        "detector": "srnet",
        "detector_source": "brijeshiitg/Pytorch-implementation-of-SRNet",
        "detector_commit": SRNET_REFERENCE_COMMIT,
        "manifest": str(Path(manifest).resolve()),
        "seeds": seeds,
        "summary": summarize(runs),
    }
    config = {
        "detector": "srnet",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seeds": seeds,
        "device": device or "auto",
        "reference_root": str(Path(reference_root).resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(manifest, run_dir / "dataset_manifest.jsonl")
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result["summary"])
    print(f"Results folder: {run_dir.resolve()}")
    return run_dir


def print_summary(summary: dict[str, object]) -> None:
    print("\nCross-domain summary")
    print("direction       accuracy       balanced acc   F1")
    print("----------------------------------------------------")
    for direction in DIRECTIONS:
        metrics = summary[direction]
        print(
            f"{direction:<15} "
            f"{metrics['accuracy_mean']:.3f} +/- {metrics['accuracy_std']:.3f}   "
            f"{metrics['balanced_accuracy_mean']:.3f} +/- "
            f"{metrics['balanced_accuracy_std']:.3f}   "
            f"{metrics['f1_mean']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.steganalysis.experiments",
        description="Pair-safe cross-domain steganalysis.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector", choices=["srnet"], default="srnet")
    parser.add_argument("--protocol", choices=["full"], default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    parser.add_argument("--output", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_srnet_protocol(
        manifest=args.manifest,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output=args.output,
        reference_root=args.srnet_root,
        device=args.device,
    )
