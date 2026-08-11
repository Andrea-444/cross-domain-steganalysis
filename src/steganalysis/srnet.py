"""Adapter around the pinned external SRNet reference implementation."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


class ManifestPairDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        domain: str,
        split: str,
        augment: bool = False,
    ) -> None:
        pairs = [
            record
            for record in records
            if record["domain"] == domain and record["split"] == split
        ]
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
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = initialize_srnet(reference_root, device)
    train_data = ManifestPairDataset(records, train_domain, "train", augment=True)
    val_data = ManifestPairDataset(records, train_domain, "val", augment=False)
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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
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

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                loss = criterion(model(images), labels)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
        average_val_loss = val_loss / max(val_total, 1)
        print(
            f"domain={train_domain} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={train_loss / max(train_total, 1):.4f} "
            f"train_acc={train_correct / max(train_total, 1):.3f} "
            f"val_loss={average_val_loss:.4f}"
        )
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, device


@torch.no_grad()
def predict_srnet(
    model,
    records: list[dict[str, object]],
    test_domain: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    dataset = ManifestPairDataset(records, test_domain, "test", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels_out: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    model.eval()
    for images, labels, _ in loader:
        output = model(images.to(device))
        probability = output.exp()[:, 1]
        predictions.extend(output.argmax(dim=1).cpu().tolist())
        probabilities.extend(probability.cpu().tolist())
        labels_out.extend(labels.tolist())
    return labels_out, predictions, probabilities
