"""Ask the actual detector whether it can tell the cover domains apart.

`src/datasets/audit.py` answers this with three hand-crafted features. That is
a useful cheap screen, but it cannot rule out differences a deep network would
find. This module runs the same question through SRNet itself, on clean covers
only, with the domain as the label.

Interpreting the result needs care. A near-chance score is only evidence that
the domains are clean if SRNet had enough data to detect a difference that
*does* exist. Always run a cohort known to be separable as a positive control:
if the probe cannot detect that one either, the null is uninformative rather
than reassuring.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.logging_config import ProgressReporter, configure_logging, get_logger
from src.datasets.manifest import read_manifest
from src.evaluation.metrics import calculate_metrics
from src.steganalysis.srnet import (
    DEFAULT_SRNET_ROOT,
    SRNET_REFERENCE_COMMIT,
    initialize_srnet,
    resolve_device,
)


DOMAIN_LABELS = {"real": 0, "ai": 1}


LOGGER = get_logger(__name__)


@dataclass
class ProbeOutcome:
    model: object
    device: "torch.device"
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_state: dict | None = None


class DomainDataset(Dataset):
    """Clean covers labelled by domain: real=0, ai=1. No stego images."""

    def __init__(self, records: list[dict[str, object]], split: str) -> None:
        rows = [
            record
            for record in records
            if str(record.get("split")) == split
            and str(record.get("domain")) in DOMAIN_LABELS
        ]
        if not rows:
            raise ValueError(f"No clean records for split={split}.")
        self.samples = [
            (
                Path(str(record["clean_path"])),
                DOMAIN_LABELS[str(record["domain"])],
                str(record["image_id"]),
            )
            for record in rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, image_id = self.samples[index]
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32)
        if array.shape != (256, 256):
            raise ValueError(f"Expected 256x256 grayscale, got {array.shape}: {path}")
        tensor = torch.from_numpy(array).unsqueeze(0) / 255.0
        return tensor, torch.tensor(label, dtype=torch.long), image_id


def train_domain_model(
    records: list[dict[str, object]],
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reference_root: str | Path = DEFAULT_SRNET_ROOT,
    device_name: str | None = None,
) -> ProbeOutcome:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_name)
    model = initialize_srnet(reference_root, device)
    train_loader = DataLoader(
        DomainDataset(records, "train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        DomainDataset(records, "val"), batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        progress = ProgressReporter(
            LOGGER, total=len(train_loader.dataset), label=f"probe epoch {epoch}"
        )
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            progress.advance(labels.size(0))
        progress.finish()

        model.eval()
        val_loss = 0.0
        val_seen = 0
        val_labels: list[int] = []
        val_predictions: list[int] = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item() * labels.size(0)
                val_seen += labels.size(0)
                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(output.argmax(dim=1).cpu().tolist())
        average_val_loss = val_loss / max(val_seen, 1)
        metrics = calculate_metrics(val_labels, val_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(seen, 1),
                "val_loss": average_val_loss,
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
        LOGGER.info(
            f"  seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={train_loss / max(seen, 1):.4f} "
            f"val_loss={average_val_loss:.4f} "
            f"val_bal_acc={metrics['balanced_accuracy']:.3f}"
        )
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return ProbeOutcome(
        model=model,
        device=device,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_state=best_state,
    )


@torch.no_grad()
def predict_domain(
    model, records: list[dict[str, object]], batch_size: int, device: torch.device
):
    loader = DataLoader(
        DomainDataset(records, "test"), batch_size=batch_size, shuffle=False
    )
    labels_out: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    image_ids: list[str] = []
    model.eval()
    for images, labels, batch_ids in loader:
        output = model(images.to(device))
        predictions.extend(output.argmax(dim=1).cpu().tolist())
        probabilities.extend(output.exp()[:, 1].cpu().tolist())
        labels_out.extend(labels.tolist())
        image_ids.extend(list(batch_ids))
    return labels_out, predictions, probabilities, image_ids


def run_domain_probe(
    manifest: str | Path,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output: str | Path,
    reference_root: str | Path = DEFAULT_SRNET_ROOT,
    device: str | None = None,
) -> Path:
    records = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    runs: list[dict[str, object]] = []
    resolved_device: str | None = None

    for seed in seeds:
        LOGGER.info(f"\nDomain probe seed {seed}")
        outcome = train_domain_model(
            records=records,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            reference_root=reference_root,
            device_name=device,
        )
        resolved_device = str(outcome.device)
        labels, predictions, probabilities, image_ids = predict_domain(
            model=outcome.model,
            records=records,
            batch_size=batch_size,
            device=outcome.device,
        )
        metrics = calculate_metrics(labels, predictions, probabilities)
        runs.append(
            {
                "seed": seed,
                "metrics": metrics,
                "selected_epoch": outcome.best_epoch,
                "best_val_loss": outcome.best_val_loss,
                "history": outcome.history,
            }
        )
        predictions_path = output / "predictions" / f"seed{seed}.jsonl"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as handle:
            for image_id, label, prediction, probability in zip(
                image_ids, labels, predictions, probabilities
            ):
                handle.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "true_domain": "ai" if label == 1 else "real",
                            "predicted_domain": "ai" if prediction == 1 else "real",
                            "p_ai": float(probability),
                            "correct": label == prediction,
                        }
                    )
                    + "\n"
                )

    def aggregate(name: str) -> dict[str, float]:
        values = [
            float(run["metrics"][name])
            for run in runs
            if run["metrics"].get(name) is not None
        ]
        if not values:
            return {}
        return {f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))}

    summary = {
        "probe": "srnet-domain-separability",
        "question": "Can SRNet identify the cover domain from clean covers alone?",
        "manifest": str(Path(manifest).resolve()),
        "seeds": seeds,
        "runs": runs,
        "aggregate": {
            **aggregate("balanced_accuracy"),
            **aggregate("accuracy"),
            **aggregate("roc_auc"),
        },
        "interpretation": (
            "0.5 balanced accuracy means SRNet cannot separate the domains. "
            "A near-chance result is only meaningful alongside a positive "
            "control cohort that the same probe does detect."
        ),
        "provenance": {
            "detector_commit": SRNET_REFERENCE_COMMIT,
            "device": resolved_device or (device or "auto"),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "platform": platform.platform(),
            "started_at": started_at,
            "duration_seconds": time.time() - started,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("\nSRNet domain separability")
    for key, value in summary["aggregate"].items():
        LOGGER.info(f"  {key:<28} {value:.3f}")
    LOGGER.info(f"  Results: {output.resolve()}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.steganalysis.domain_probe",
        description="Test whether SRNet can identify cover domain without stego.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    run_domain_probe(
        manifest=args.manifest,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output=args.output,
        reference_root=args.srnet_root,
        device=args.device,
    )
