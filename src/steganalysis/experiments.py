from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.logging_config import configure_logging, get_logger
from src.datasets.manifest import read_manifest, validate_stego_manifest
from src.evaluation.metrics import calculate_metrics, summarize_directions
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


LOGGER = get_logger(__name__)


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    return summarize_directions(runs, DIRECTIONS)


def validate_pair_splits(records: list[dict[str, object]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        image_id = str(record["image_id"])
        split = str(record["split"])
        previous = seen.setdefault(image_id, split)
        if previous != split:
            raise ValueError(f"Pair leakage: {image_id} appears in {previous} and {split}.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_counts(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[str(record["domain"])][str(record["split"])] += 1
    return {domain: dict(splits) for domain, splits in counts.items()}


def embedding_condition(records: list[dict[str, object]]) -> dict[str, object]:
    """Read the embedding condition off the manifest rather than trusting a flag.

    ``embedding_seed`` is derived per image by the embedding step, so listing
    every distinct value would record noise instead of provenance. A varying
    seed is summarised; the run-level master seed lives in the stego run's own
    ``config.json``.
    """
    condition: dict[str, object] = {}
    for key in ("algorithm", "payload_bpp", "embedding_seed"):
        values = {record.get(key) for record in records if record.get(key) is not None}
        if len(values) == 1:
            condition[key] = values.pop()
        elif key == "embedding_seed":
            condition[key] = {"per_image": True, "distinct": len(values)}
        else:
            condition[key] = sorted(map(str, values))
    return condition


def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "numpy", "sklearn", "PIL", "conseal"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    return versions


def write_predictions(
    path: Path,
    image_ids: list[str],
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
    domain: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for image_id, label, prediction, probability in zip(
            image_ids, labels, predictions, probabilities
        ):
            handle.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "domain": domain,
                        "split": "test",
                        "label": int(label),
                        "prediction": int(prediction),
                        "probability": float(probability),
                        "correct": int(label) == int(prediction),
                    }
                )
                + "\n"
            )


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
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    resolved_device: str | None = None

    for seed in seeds:
        outcomes = {}
        for train_domain in ("real", "ai"):
            outcome = train_srnet(
                records=records,
                train_domain=train_domain,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                reference_root=reference_root,
                device_name=device,
            )
            outcomes[train_domain] = outcome
            resolved_device = str(outcome.device)
            history_path = run_dir / "history" / f"seed{seed}_{train_domain}.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "train_domain": train_domain,
                        "epochs": outcome.history,
                        "selection": {
                            "criterion": "val_loss",
                            "epoch": outcome.best_epoch,
                            "val_loss": outcome.best_val_loss,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if outcome.best_state is not None:
                checkpoint_path = run_dir / "checkpoints" / f"seed{seed}_{train_domain}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": outcome.best_state,
                        "seed": seed,
                        "train_domain": train_domain,
                        "selected_epoch": outcome.best_epoch,
                        "selection_criterion": "val_loss",
                        "val_loss": outcome.best_val_loss,
                        "detector_commit": SRNET_REFERENCE_COMMIT,
                    },
                    checkpoint_path,
                )

        direction_results: dict[str, object] = {}
        for name, (train_domain, test_domain) in DIRECTIONS.items():
            outcome = outcomes[train_domain]
            labels, predictions, probabilities, image_ids = predict_srnet(
                model=outcome.model,
                records=records,
                test_domain=test_domain,
                batch_size=batch_size,
                device=outcome.device,
            )
            direction_results[name] = calculate_metrics(labels, predictions, probabilities)
            write_predictions(
                run_dir / "predictions" / f"seed{seed}_{name}.jsonl",
                image_ids, labels, predictions, probabilities, test_domain,
            )
        runs.append({"seed": seed, "directions": direction_results})

    repository_commit, repository_dirty = git_state()
    result = {
        "detector": "srnet",
        "detector_source": "brijeshiitg/Pytorch-implementation-of-SRNet",
        "detector_commit": SRNET_REFERENCE_COMMIT,
        "manifest": str(Path(manifest).resolve()),
        "seeds": seeds,
        "runs": runs,
        "summary": summarize(runs),
        "provenance": {
            "manifest_sha256": file_sha256(manifest),
            "dataset_counts": dataset_counts(records),
            "embedding": embedding_condition(records),
            "detector_commit": SRNET_REFERENCE_COMMIT,
            "repository_commit": repository_commit,
            "repository_dirty": repository_dirty,
            "device": resolved_device or (device or "auto"),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "dependencies": dependency_versions(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.time() - started,
        },
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
    LOGGER.info(f"Results folder: {run_dir.resolve()}")
    return run_dir


def print_summary(summary: dict[str, object]) -> None:
    LOGGER.info("\nCross-domain summary")
    LOGGER.info("direction       accuracy       balanced acc   F1")
    LOGGER.info("----------------------------------------------------")
    for direction in DIRECTIONS:
        metrics = summary[direction]
        LOGGER.info(
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
    configure_logging()
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
