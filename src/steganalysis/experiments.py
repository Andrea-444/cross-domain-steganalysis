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
from importlib.metadata import PackageNotFoundError, version
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
from src.tracking.naming import (
    manifest_condition,
    run_group,
    run_name,
    run_role,
    run_tags,
)
from src.tracking.wandb import start_experiment_run


# Ordered by the accepted experiment priority: same-domain answers research
# question 1, mixed-domain separates distribution shift from a real difference
# in detectability, and cross-domain is interpreted last and against the mixed
# result. See notes/Decisions.md, 2026-10-10.
DIRECTIONS = {
    "real_to_real": ("real", "real"),
    "ai_to_ai": ("ai", "ai"),
    "mixed_to_real": ("mixed", "real"),
    "mixed_to_ai": ("mixed", "ai"),
    "real_to_ai": ("real", "ai"),
    "ai_to_real": ("ai", "real"),
}

TRAIN_DOMAINS = ("real", "ai", "mixed")


LOGGER = get_logger(__name__)


def summarize(
    runs: list[dict[str, object]],
    directions: list[str] | None = None,
) -> dict[str, object]:
    return summarize_directions(runs, directions or list(DIRECTIONS))


def select_directions(names: list[str] | None) -> list[str]:
    """Keep the accepted priority order whatever order the request gives.

    The order is part of the protocol: same-domain is read first, mixed-domain
    second, and cross-domain last and against the mixed result. A stage that
    asks for the directions in a different order still gets them in this one.
    """
    if not names:
        return list(DIRECTIONS)
    unknown = sorted({name for name in names if name not in DIRECTIONS})
    if unknown:
        raise ValueError(
            f"Unknown direction(s): {', '.join(unknown)}. "
            f"Known directions: {', '.join(DIRECTIONS)}."
        )
    requested = set(names)
    return [name for name in DIRECTIONS if name in requested]


def required_train_domains(directions: list[str]) -> list[str]:
    """Train only the domains that the selected directions test.

    Each train domain supplies two directions, so a stage that takes both
    directions of one domain costs the same training as a stage that takes
    one of them. A stage that splits the two retrains the same model twice.
    """
    needed = {DIRECTIONS[name][0] for name in directions}
    return [domain for domain in TRAIN_DOMAINS if domain in needed]


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
    distributions = {
        "torch": "torch",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "conseal": "conseal",
    }
    for name, distribution in distributions.items():
        try:
            versions[name] = version(distribution)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def accelerator_state() -> dict[str, object]:
    """Record the GPU driver beside the library versions.

    A driver upgrade on 2026-10-10 left the GPU machine with kernel module
    595.84 in memory and 595.99.02 on disk. The runs before and after that date
    therefore ran in different environments, and nothing in the provenance said
    so. The library version of torch does not change when a driver does.

    `nvml_available` is recorded because a broken NVML is the visible symptom
    of a mismatch: CUDA arithmetic continues to work, and the management
    library stops.
    """
    state: dict[str, object] = {
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "kernel_driver": None,
        "nvml_available": None,
        "device_name": None,
    }
    # Linux exposes the running kernel module here. The file is absent on
    # macOS and on a machine with no NVIDIA driver, which is not an error.
    version_file = Path("/proc/driver/nvidia/version")
    try:
        if version_file.exists():
            first_line = version_file.read_text(encoding="utf-8").splitlines()[0]
            state["kernel_driver"] = first_line.strip()
    except OSError:
        pass
    state["mps_available"] = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if not torch.cuda.is_available():
        return state
    try:
        state["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import pynvml
    except ImportError:
        # No reporting library present. That is not the same as a broken one,
        # so the field stays None rather than False.
        return state
    try:
        pynvml.nvmlInit()
        state["nvml_available"] = True
        pynvml.nvmlShutdown()
    except Exception:
        state["nvml_available"] = False
    return state


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
    directions: list[str] | None = None,
    wandb_mode: str = "disabled",
    wandb_project: str = "stego-cross-domain",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_upload_artifact: bool = False,
) -> Path:
    records = read_manifest(manifest)
    validate_stego_manifest(records)
    validate_pair_splits(records)
    selected_directions = select_directions(directions)
    train_domains = required_train_domains(selected_directions)
    run_dir = timestamped_results_dir(output, "srnet")
    runs: list[dict[str, object]] = []
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    resolved_device: str | None = None
    experiment_config = {
        "detector": "srnet",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seeds": seeds,
        "device": device or "auto",
        "reference_root": str(Path(reference_root).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": file_sha256(manifest),
        "dataset_counts": dataset_counts(records),
        "embedding": embedding_condition(records),
        "directions": selected_directions,
        "train_domains": train_domains,
    }
    # The name, tags and group come from the run itself rather than from a
    # flag, so a run in W&B states the condition it actually used. The
    # embedding and the preprocessing live in the manifest chain, which no
    # command argument repeats.
    condition = manifest_condition(manifest)
    role = run_role(seeds, len(selected_directions), len(DIRECTIONS))
    derived_name = run_name(
        role=role,
        train_domains=train_domains,
        condition=condition,
        learning_rate=learning_rate,
        seeds=seeds,
    )
    derived_tags = run_tags(
        role=role,
        detector="srnet",
        train_domains=train_domains,
        directions=selected_directions,
        condition=condition,
        learning_rate=learning_rate,
        seeds=seeds,
        extra=wandb_tags,
    )
    experiment_config["condition"] = condition
    experiment_config["role"] = role
    tracker = None
    tracking: dict[str, object] = {"status": "disabled"}
    if wandb_mode != "disabled":
        try:
            tracker = start_experiment_run(
                mode=wandb_mode,
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or derived_name,
                config=experiment_config,
                tags=derived_tags,
                upload_artifact=wandb_upload_artifact,
                group=run_group(condition, experiment_config["manifest_sha256"]),
                job_type=role,
            )
            tracking = {
                "status": "active",
                "mode": wandb_mode,
                "project": wandb_project,
            }
        except Exception as error:
            tracking = {"status": "failed", "error": str(error)}
            LOGGER.warning("W&B initialization failed; continuing locally: %s", error)

    for seed in seeds:
        outcomes = {}
        for train_domain in train_domains:
            outcome = train_srnet(
                records=records,
                train_domain=train_domain,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                reference_root=reference_root,
                device_name=device,
                epoch_callback=(
                    lambda record, current_seed=seed, current_domain=train_domain: (
                        tracker.log_epoch(current_seed, current_domain, record)
                    )
                    if tracker is not None
                    else None
                ),
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
        for name in selected_directions:
            train_domain, test_domain = DIRECTIONS[name]
            outcome = outcomes[train_domain]
            labels, predictions, probabilities, image_ids = predict_srnet(
                model=outcome.model,
                records=records,
                test_domain=test_domain,
                batch_size=batch_size,
                device=outcome.device,
            )
            metrics = calculate_metrics(labels, predictions, probabilities)
            direction_results[name] = metrics
            if tracker is not None:
                tracker.log_evaluation(seed, name, metrics)
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
        "summary": summarize(runs, selected_directions),
        "tracking": tracking,
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
            "accelerator": accelerator_state(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.time() - started,
        },
    }
    config = {
        **experiment_config,
        "tracking": {
            "mode": wandb_mode,
            "project": wandb_project,
            "entity": wandb_entity,
            "run_name": wandb_run_name or derived_name,
            "tags": derived_tags,
            "group": run_group(condition, experiment_config["manifest_sha256"]),
            "job_type": role,
            "upload_artifact": wandb_upload_artifact,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(manifest, run_dir / "dataset_manifest.jsonl")
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if tracker is not None:
        result["tracking"] = tracker.finish(result, run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print_summary(result["summary"], selected_directions)
    LOGGER.info(f"Results folder: {run_dir.resolve()}")
    return run_dir


def print_summary(
    summary: dict[str, object],
    directions: list[str] | None = None,
) -> None:
    LOGGER.info("\nCross-domain summary")
    LOGGER.info("direction       accuracy       balanced acc   F1")
    LOGGER.info("----------------------------------------------------")
    for direction in directions or list(DIRECTIONS):
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
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=list(DIRECTIONS),
        default=None,
        help=(
            "Run a subset of the directions, for a staged run. The default is "
            "all six. Keep both directions of one train domain in the same "
            "stage: they share the trained model, and their paired gap needs "
            "both of them."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "online", "offline"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="stego-cross-domain")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-upload-artifact", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    run_srnet_protocol(
        manifest=args.manifest,
        seeds=args.seeds,
        directions=args.directions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output=args.output,
        reference_root=args.srnet_root,
        device=args.device,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_upload_artifact=args.wandb_upload_artifact,
    )
