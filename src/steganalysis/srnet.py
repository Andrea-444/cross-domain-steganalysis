"""Adapter around the pinned external SRNet reference implementation."""

from __future__ import annotations

import importlib
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    limit: int | None = None,
    mixed_pairs_per_domain: int | None = None,
) -> list[dict[str, object]]:
    """Return the covers one detector trains on, for one domain and split.

    The `mixed` domain holds half its covers from each single domain, and the
    **same total** as a single-domain set. A mixed detector trained on the
    union of both domains would see twice the data, and any advantage it showed
    could come from data volume rather than from domain coverage.

    `limit` caps the number of pairs. It exists to answer one question: how
    much of a domain's result comes from the quantity of training data. The
    mixed detector holds half the pairs of a specialist in each domain, so a
    specialist trained at the mixed count separates "this domain needs more
    data" from "the mixed detector gives up something else".

    ``mixed_pairs_per_domain`` is an explicit sensitivity condition. It gives a
    mixed detector that many pairs from *each* cover domain, so the total is
    twice the requested value. It is separate from ``limit`` because the two
    options answer different questions and must not be confused.

    Selection is deterministic for a seed, so a repeated run trains on the same
    covers. A smaller limit holds a subset of a larger one, because both draw
    from the same seeded order.
    """
    if domain not in TRAINING_DOMAINS:
        raise ValueError(f"domain must be one of {TRAINING_DOMAINS}, got {domain!r}.")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}.")
    if mixed_pairs_per_domain is not None and mixed_pairs_per_domain < 1:
        raise ValueError(
            "mixed_pairs_per_domain must be at least 1, got "
            f"{mixed_pairs_per_domain}."
        )
    if limit is not None and mixed_pairs_per_domain is not None:
        raise ValueError("limit and mixed_pairs_per_domain cannot be used together.")
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
        if mixed_pairs_per_domain is not None:
            raise ValueError("mixed_pairs_per_domain applies only to domain='mixed'.")
        chosen = list(by_domain[domain])
        return _take(chosen, limit, f"limit-{domain}-{seed}-{split}")

    if mixed_pairs_per_domain is not None:
        available = {name: len(rows) for name, rows in by_domain.items()}
        short = {
            name: count
            for name, count in available.items()
            if count < mixed_pairs_per_domain
        }
        if short:
            detail = ", ".join(f"{name}={count}" for name, count in short.items())
            raise ValueError(
                f"Requested {mixed_pairs_per_domain} mixed training pairs per domain, "
                f"but only {detail} are available for split={split!r}."
            )
        generator = random.Random(f"mixed-{seed}-{split}")
        chosen = []
        for name in ("real", "ai"):
            chosen.extend(generator.sample(by_domain[name], mixed_pairs_per_domain))
        chosen.sort(key=lambda record: str(record["image_id"]))
        return chosen

    target = min(len(by_domain["real"]), len(by_domain["ai"]))
    if limit is not None:
        target = min(target, limit)
    half = target // 2
    generator = random.Random(f"mixed-{seed}-{split}")
    chosen = []
    for name in ("real", "ai"):
        chosen.extend(generator.sample(by_domain[name], min(half, len(by_domain[name]))))
    chosen.sort(key=lambda record: str(record["image_id"]))
    return chosen


def _take(
    rows: list[dict[str, object]], limit: int | None, salt: str
) -> list[dict[str, object]]:
    """Keep `limit` rows, chosen by a seeded shuffle rather than by position.

    Taking the first rows of a sorted list would bias the subset, because the
    order comes from the image identifier. A seeded shuffle keeps the choice
    deterministic and representative.
    """
    if limit is None or limit >= len(rows):
        return rows
    order = list(rows)
    random.Random(salt).shuffle(order)
    kept = order[:limit]
    kept.sort(key=lambda record: str(record["image_id"]))
    return kept


class ManifestPairDataset(Dataset):
    """Cover and stego images from a manifest, flat or paired.

    In ``paired`` mode one index gives a cover and its own stego, and both get
    the same augmentation draw. The reference implementation does the same, and
    the pair constraint is what makes the batch informative: matched content on
    both sides, so the only systematic difference inside a batch is the
    embedding. A flat shuffle puts a cover and its stego in different batches
    and gives each a separate rotation and flip, which removes that structure.

    Evaluation uses the flat mode, because a prediction is per image.
    """

    def __init__(
        self,
        records: list[dict[str, object]],
        domain: str,
        split: str,
        augment: bool = False,
        preselected: bool = False,
        paired: bool = False,
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
        self.pairs: list[tuple[Path, Path, str]] = []
        self.samples: list[tuple[Path, int, str]] = []
        for record in pairs:
            image_id = str(record["image_id"])
            clean_path = Path(str(record["clean_path"]))
            stego_path = Path(str(record["stego_path"]))
            self.pairs.append((clean_path, stego_path, image_id))
            self.samples.append((clean_path, 0, image_id))
            self.samples.append((stego_path, 1, image_id))
        self.augment = augment
        self.paired = paired

    def __len__(self) -> int:
        return len(self.pairs) if self.paired else len(self.samples)

    def _read(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32)
        if array.shape != (256, 256):
            raise ValueError(f"SRNet expects 256x256 grayscale images, got {array.shape}: {path}")
        return array

    def _draw(self) -> tuple[int, bool]:
        if not self.augment:
            return 0, False
        return random.randint(0, 3), random.random() < 0.5

    @staticmethod
    def _to_tensor(array: np.ndarray, rotation: int, flip: bool) -> torch.Tensor:
        array = np.rot90(array, rotation).copy()
        if flip:
            array = np.fliplr(array).copy()
        return torch.from_numpy(array).unsqueeze(0) / 255.0

    def __getitem__(self, index: int):
        if self.paired:
            clean_path, stego_path, image_id = self.pairs[index]
            # One draw for the pair. Two draws would rotate a cover and its
            # stego differently, and the pair would no longer be comparable.
            rotation, flip = self._draw()
            return (
                self._to_tensor(self._read(clean_path), rotation, flip),
                self._to_tensor(self._read(stego_path), rotation, flip),
                image_id,
            )
        path, label, image_id = self.samples[index]
        rotation, flip = self._draw()
        tensor = self._to_tensor(self._read(path), rotation, flip)
        return tensor, torch.tensor(label, dtype=torch.long), image_id


def paired_batch(
    clean_images: torch.Tensor,
    stego_images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join one paired batch into images and labels, covers first."""
    images = torch.cat([clean_images, stego_images]).to(device)
    labels = torch.cat(
        [
            torch.zeros(clean_images.size(0), dtype=torch.long),
            torch.ones(stego_images.size(0), dtype=torch.long),
        ]
    ).to(device)
    return images, labels


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
    """Initialize SRNet with the reference implementation's own scheme.

    The reference applies `weights_init` from its utils module: Xavier uniform
    on the convolutions, and normal(0, 0.01) on the fully connected layer. We
    copy the rules rather than import them, because that module runs argparse
    when it loads.

    The fully connected layer is the part that matters. Xavier on a 512-to-2
    layer gives a standard deviation near 0.06, so the network starts with
    large logits and a loss above ln(2). The fastest way down is then to make
    the output constant, and a measurement showed the network doing exactly
    that: the output difference between a cover and its own stego fell from
    1.3e-3 to 6.7e-5 in 200 steps, at every learning rate tested. A small
    fully connected layer starts the network near ln(2), so that shortcut is
    not available and the loss can only fall by separating the two classes.
    """
    srnet_class = load_srnet_class(reference_root)
    model = srnet_class().to(device)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.2)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
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
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    train_pairs: int | None = None,
    mixed_pairs_per_domain: int | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(device_name)
    model = initialize_srnet(reference_root, device)
    train_data = ManifestPairDataset(
        select_training_records(
            records,
            train_domain,
            "train",
            seed,
            train_pairs,
            mixed_pairs_per_domain=(
                mixed_pairs_per_domain if train_domain == "mixed" else None
            ),
        ),
        train_domain, "train", augment=True, preselected=True, paired=True,
    )
    val_data = ManifestPairDataset(
        select_training_records(records, train_domain, "val", seed),
        train_domain, "val", augment=False, preselected=True, paired=True,
    )
    # batch_size counts images, as it did before. A paired batch holds half as
    # many covers and the same number of stego twins, so the images for each
    # optimizer step stay the same and only their composition changes.
    pairs_per_batch = max(1, batch_size // 2)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=pairs_per_batch,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=pairs_per_batch, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    LOGGER.info(
        "Training on domain=%s seed=%s: %d epochs over %d images "
        "(%d pairs, %d per batch), device=%s.",
        train_domain, seed, epochs, 2 * len(train_data),
        len(train_data), pairs_per_batch, device,
    )
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        progress = ProgressReporter(
            LOGGER, total=2 * len(train_data), label=f"{train_domain} epoch {epoch}"
        )
        for clean_images, stego_images, _ in train_loader:
            images, labels = paired_batch(clean_images, stego_images, device)
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
            for clean_images, stego_images, _ in val_loader:
                images, labels = paired_batch(clean_images, stego_images, device)
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
        if epoch_callback is not None:
            epoch_callback(dict(epoch_record))
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
