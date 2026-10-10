"""Adapter around the pinned external SRNet reference implementation."""

from __future__ import annotations

import importlib
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import calculate_metrics
from src.logging_config import ProgressReporter, get_logger


LOGGER = get_logger(__name__)


@dataclass
class TrainingOutcome:
    """Everything one training run produces beyond the fitted model."""

    model: object
    device: "torch.device"
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_state: dict | None = None


DEFAULT_SRNET_ROOT = Path("external/srnet-reference")
SRNET_REFERENCE_COMMIT = "2c838cf7d26dd0b8e45eda79ebf103ca312c0355"


def load_srnet_class(reference_root: str | Path = DEFAULT_SRNET_ROOT):
    reference_root = Path(reference_root).resolve()
    model_file = reference_root / "model" / "model.py"
    if not model_file.exists():
        raise RuntimeError(
            "SRNet reference implementation is missing. Clone "
            "https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet "
            f"at commit {SRNET_REFERENCE_COMMIT} into {reference_root}."
        )
    root_string = str(reference_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    module = importlib.import_module("model.model")
    return module.Srnet


TRAINING_DOMAINS = ("real", "ai", "mixed")


def select_training_records(
    records: list[dict[str, object]],
    domain: str,
    split: str,
    seed: int,
) -> list[dict[str, object]]:
    """Return the covers one detector trains on, for one domain and split.

    The `mixed` domain holds half its covers from each single domain, and the
    **same total** as a single-domain set. A mixed detector trained on the
    union of both domains would see twice the data, and any advantage it showed
    could come from data volume rather than from domain coverage.

    Selection is deterministic for a seed, so a repeated run trains on the same
    covers.
    """
    if domain not in TRAINING_DOMAINS:
        raise ValueError(f"domain must be one of {TRAINING_DOMAINS}, got {domain!r}.")
    by_domain: dict[str, list[dict[str, object]]] = {}
    for name in ("real", "ai"):
        rows = [
            record
            for record in records
            if str(record["domain"]) == name and str(record["split"]) == split
        ]
        rows.sort(key=lambda record: str(record["image_id"]))
        by_domain[name] = rows

    if domain != "mixed":
        return list(by_domain[domain])

    target = min(len(by_domain["real"]), len(by_domain["ai"]))
    half = target // 2
    generator = random.Random(f"mixed-{seed}-{split}")
    chosen: list[dict[str, object]] = []
    for name in ("real", "ai"):
        chosen.extend(generator.sample(by_domain[name], min(half, len(by_domain[name]))))
    chosen.sort(key=lambda record: str(record["image_id"]))
    return chosen


class ManifestPairDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        domain: str,
        split: str,
        augment: bool = False,
        preselected: bool = False,
    ) -> None:
        pairs = (
            list(records)
            if preselected
            else [
                record
                for record in records
                if record["domain"] == domain and record["split"] == split
            ]
        )
        if not pairs:
            raise ValueError(f"No records for domain={domain}, split={split}.")
        self.samples: list[tuple[Path, int, str]] = []
        for record in pairs:
            image_id = str(record["image_id"])
            self.samples.append((Path(str(record["clean_path"])), 0, image_id))
            self.samples.append((Path(str(record["stego_path"])), 1, image_id))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, image_id = self.samples[index]
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32)
        if array.shape != (256, 256):
            raise ValueError(f"SRNet expects 256x256 grayscale images, got {array.shape}: {path}")
        if self.augment:
            rotation = random.randint(0, 3)
            array = np.rot90(array, rotation).copy()
            if random.random() < 0.5:
                array = np.fliplr(array).copy()
        tensor = torch.from_numpy(array).unsqueeze(0) / 255.0
        return tensor, torch.tensor(label, dtype=torch.long), image_id


def resolve_device(device_name: str | None) -> torch.device:
    """Pick the fastest available device: explicit override, then CUDA, MPS, CPU.

    Apple Silicon exposes MPS rather than CUDA. On this project's reference
    machine an SRNet training step runs roughly nineteen times faster on MPS
    than on CPU, so falling straight back to CPU makes full runs impractical.
    """
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_srnet(reference_root: str | Path, device: torch.device):
    srnet_class = load_srnet_class(reference_root)
    model = srnet_class().to(device)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def train_srnet(
    records: list[dict[str, object]],
    train_domain: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reference_root: str | Path = DEFAULT_SRNET_ROOT,
    device_name: str | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(device_name)
    model = initialize_srnet(reference_root, device)
    train_data = ManifestPairDataset(
        select_training_records(records, train_domain, "train", seed),
        train_domain, "train", augment=True, preselected=True,
    )
    val_data = ManifestPairDataset(
        select_training_records(records, train_domain, "val", seed),
        train_domain, "val", augment=False, preselected=True,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    LOGGER.info(
        "Training on domain=%s seed=%s: %d epochs over %d images, device=%s.",
        train_domain, seed, epochs, len(train_data), device,
    )
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        progress = ProgressReporter(
            LOGGER, total=len(train_data), label=f"{train_domain} epoch {epoch}"
        )
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (output.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            progress.advance(labels.size(0))
        progress.finish()

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_labels: list[int] = []
        val_predictions: list[int] = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(output.argmax(dim=1).cpu().tolist())
        average_val_loss = val_loss / max(val_total, 1)
        val_metrics = calculate_metrics(val_labels, val_predictions)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": average_val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": time.time() - epoch_started,
        }
        history.append(epoch_record)
        LOGGER.info(
            f"domain={train_domain} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_acc={epoch_record['train_accuracy']:.3f} "
            f"val_loss={average_val_loss:.4f} "
            f"val_acc={epoch_record['val_accuracy']:.3f}"
        )
        # Selection stays on validation loss: changing the criterion would be a
        # protocol change, so the accuracy metrics above are recorded only.
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
    return TrainingOutcome(
        model=model,
        device=device,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_state=best_state,
    )


@torch.no_grad()
def predict_srnet(
    model,
    records: list[dict[str, object]],
    test_domain: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int], list[float], list[str]]:
    dataset = ManifestPairDataset(records, test_domain, "test", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels_out: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    image_ids: list[str] = []
    model.eval()
    for images, labels, batch_ids in loader:
        output = model(images.to(device))
        # The reference model ends in LogSoftmax, so exp() gives P(stego).
        probability = output.exp()[:, 1]
        predictions.extend(output.argmax(dim=1).cpu().tolist())
        probabilities.extend(probability.cpu().tolist())
        labels_out.extend(labels.tolist())
        image_ids.extend(list(batch_ids))
    return labels_out, predictions, probabilities, image_ids
