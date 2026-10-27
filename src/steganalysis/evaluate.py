"""Evaluate saved SRNet checkpoints on a new paired test manifest."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.datasets.manifest import read_manifest, validate_stego_manifest
from src.evaluation.metrics import calculate_metrics, summarize_directions
from src.logging_config import configure_logging, get_logger
from src.steganalysis.experiments import (
    accelerator_state,
    dataset_counts,
    dependency_versions,
    embedding_condition,
    file_sha256,
    git_state,
    timestamped_results_dir,
    validate_pair_splits,
    write_predictions,
)
from src.steganalysis.srnet import (
    DEFAULT_SRNET_ROOT,
    SRNET_REFERENCE_COMMIT,
    initialize_srnet,
    predict_srnet,
    resolve_device,
)


LOGGER = get_logger(__name__)
CONDITION_FIELDS = (
    "train_domain",
    "source_manifest_sha256",
    "epochs",
    "batch_size",
    "learning_rate",
    "train_pairs",
    "mixed_pairs_per_domain",
    "detector_commit",
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or "checkpoint"


def _source_config(checkpoint: Path) -> tuple[Path, dict[str, object]]:
    path = checkpoint.parent.parent / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No config.json beside checkpoint run folder: {checkpoint}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    detector_commit = payload.get("detector_commit")
    if detector_commit != SRNET_REFERENCE_COMMIT:
        raise ValueError(
            f"Checkpoint {path} uses detector commit {detector_commit!r}; "
            f"expected {SRNET_REFERENCE_COMMIT!r}."
        )
    if payload.get("train_domain") not in {"real", "ai", "mixed"}:
        raise ValueError(f"Checkpoint has an invalid train_domain: {path}")
    if not isinstance(payload.get("seed"), int):
        raise ValueError(f"Checkpoint has no integer seed: {path}")
    return payload


def _condition(
    metadata: dict[str, object], source_config: dict[str, object]
) -> dict[str, object]:
    return {
        "train_domain": metadata["train_domain"],
        "source_manifest_sha256": source_config.get("manifest_sha256"),
        "epochs": source_config.get("epochs"),
        "batch_size": source_config.get("batch_size"),
        "learning_rate": source_config.get("learning_rate"),
        "train_pairs": source_config.get("train_pairs"),
        "mixed_pairs_per_domain": source_config.get("mixed_pairs_per_domain"),
        "detector_commit": metadata["detector_commit"],
    }


def _condition_key(condition: dict[str, object]) -> tuple[object, ...]:
    return tuple(condition.get(field) for field in CONDITION_FIELDS)


def _aggregate(evaluations: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[_condition_key(evaluation["training_condition"])].append(evaluation)

    conditions: list[dict[str, object]] = []
    for key, rows in grouped.items():
        direction = "evaluation"
        runs = [
            {"seed": row["seed"], "directions": {direction: row["metrics"]}}
            for row in rows
        ]
        conditions.append(
            {
                "training_condition": dict(zip(CONDITION_FIELDS, key)),
                "seeds": sorted(int(row["seed"]) for row in rows),
                "checkpoints": sorted(str(row["checkpoint"]) for row in rows),
                "summary": summarize_directions(runs, [direction])[direction],
            }
        )
    conditions.sort(key=lambda row: str(row["training_condition"]))
    return conditions


def evaluate_checkpoints(
    manifest: str | Path,
    checkpoints: list[str | Path],
    test_domain: str,
    test_set_name: str,
    batch_size: int,
    output: str | Path,
    reference_root: str | Path = DEFAULT_SRNET_ROOT,
    device_name: str | None = None,
) -> Path:
    """Evaluate checkpoints without training and preserve full provenance."""
    manifest = Path(manifest).resolve()
    checkpoint_paths = [Path(path).resolve() for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")
    if test_domain not in {"real", "ai"}:
        raise ValueError("test_domain must be 'real' or 'ai'.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    records = read_manifest(manifest)
    validate_stego_manifest(records)
    validate_pair_splits(records)
    available = [
        record
        for record in records
        if str(record["domain"]) == test_domain and str(record["split"]) == "test"
    ]
    if not available:
        raise ValueError(f"No test records for domain={test_domain!r}.")

    run_dir = timestamped_results_dir(output, f"srnet_eval_{_safe_name(test_set_name)}")
    device = resolve_device(device_name)
    started = datetime.now(timezone.utc).isoformat()
    evaluations: list[dict[str, object]] = []

    for checkpoint in checkpoint_paths:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        metadata = _load_checkpoint(checkpoint)
        source_config_path, source_config = _source_config(checkpoint)
        condition = _condition(metadata, source_config)
        model = initialize_srnet(reference_root, device)
        model.load_state_dict(metadata["state_dict"])
        model.to(device)
        labels, predictions, probabilities, image_ids = predict_srnet(
            model=model,
            records=records,
            test_domain=test_domain,
            batch_size=batch_size,
            device=device,
        )
        metrics = calculate_metrics(labels, predictions, probabilities)
        label = _safe_name(
            f"{checkpoint.parent.parent.name}-seed{metadata['seed']}-"
            f"{metadata['train_domain']}"
        )
        prediction_path = run_dir / "predictions" / f"{label}.jsonl"
        write_predictions(
            prediction_path,
            image_ids,
            labels,
            predictions,
            probabilities,
            test_domain,
        )
        evaluations.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "source_config": str(source_config_path),
                "seed": metadata["seed"],
                "train_domain": metadata["train_domain"],
                "selected_epoch": metadata.get("selected_epoch"),
                "selection_criterion": metadata.get("selection_criterion"),
                "selected_val_loss": metadata.get("val_loss"),
                "training_condition": condition,
                "metrics": metrics,
                "predictions": str(prediction_path),
            }
        )
        LOGGER.info(
            "Evaluated %s seed=%s on %s: balanced_accuracy=%.4f ROC-AUC=%.4f",
            metadata["train_domain"],
            metadata["seed"],
            test_set_name,
            float(metrics["balanced_accuracy"]),
            float(metrics["roc_auc"]),
        )

    repository_commit, repository_dirty = git_state()
    result = {
        "task": "checkpoint_evaluation",
        "test_set_name": test_set_name,
        "test_domain": test_domain,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "dataset_counts": dataset_counts(records),
        "embedding": embedding_condition(records),
        "sample_count": 2 * len(available),
        "evaluations": evaluations,
        "conditions": _aggregate(evaluations),
        "provenance": {
            "detector_commit": SRNET_REFERENCE_COMMIT,
            "repository_commit": repository_commit,
            "repository_dirty": repository_dirty,
            "device": str(device),
            "dependencies": dependency_versions(),
            "accelerator": accelerator_state(),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    config = {
        "manifest": str(manifest),
        "manifest_sha256": result["manifest_sha256"],
        "checkpoints": [str(path) for path in checkpoint_paths],
        "test_domain": test_domain,
        "test_set_name": test_set_name,
        "batch_size": batch_size,
        "device": device_name or "auto",
        "reference_root": str(Path(reference_root).resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(manifest, run_dir / "dataset_manifest.jsonl")
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation results: %s", run_dir.resolve())
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.steganalysis.evaluate",
        description="Evaluate saved SRNet checkpoints without retraining.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-domain", choices=["real", "ai"], required=True)
    parser.add_argument("--test-set-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    parser.add_argument("--output", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    evaluate_checkpoints(
        manifest=args.manifest,
        checkpoints=args.checkpoints,
        test_domain=args.test_domain,
        test_set_name=args.test_set_name,
        batch_size=args.batch_size,
        output=args.output,
        reference_root=args.srnet_root,
        device_name=args.device,
    )
