from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.legacy.model import SmallStegoCNN


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    predicted_clean: int
    predicted_stego: int


def train_model(
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    input_channels: int = 3,
    device: str | None = None,
) -> SmallStegoCNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallStegoCNN(input_channels=input_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            progress.set_postfix(loss=np.mean(losses), acc=correct / max(total, 1))

    return model


@torch.no_grad()
def evaluate_model(
    model: SmallStegoCNN,
    data_loader: DataLoader,
    device: str | None = None,
) -> Metrics:
    device = device or next(model.parameters()).device.type
    model.eval()
    model.to(device)
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in data_loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return Metrics(
        accuracy=accuracy_score(all_labels, all_preds),
        precision=precision_score(all_labels, all_preds, zero_division=0),
        recall=recall_score(all_labels, all_preds, zero_division=0),
        f1=f1_score(all_labels, all_preds, zero_division=0),
        confusion_matrix=confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist(),
        predicted_clean=all_preds.count(0),
        predicted_stego=all_preds.count(1),
    )
