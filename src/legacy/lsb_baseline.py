from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.legacy.stego_lsb import IMAGE_EXTENSIONS
from src.legacy.train import Metrics


def collect_samples(
    root: str | Path,
    category: str | None = None,
    clean_root: str | Path | None = None,
) -> list[tuple[Path, int]]:
    root = Path(root)
    samples: list[tuple[Path, int]] = []
    roots = {
        "clean": Path(clean_root) if clean_root is not None else root / "clean",
        "stego": root / "stego",
    }
    for class_name, label in (("clean", 0), ("stego", 1)):
        class_dir = roots[class_name]
        if category is not None:
            class_dir = class_dir / category
        if not class_dir.exists():
            raise ValueError(f"Missing folder: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend((path, label) for path in paths)
    if not samples:
        raise ValueError(f"No images found under {root}")
    return samples


def extract_lsb_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    features: list[float] = []
    for channel in cv2.split(image):
        low_nibble = channel & 0x0F
        hist = np.bincount(low_nibble.reshape(-1), minlength=16).astype(np.float32)
        hist /= hist.sum()
        features.extend(hist.tolist())

        bit0 = channel & 1
        features.append(float(bit0.mean()))
        features.append(float(np.mean(bit0[:, 1:] != bit0[:, :-1])))
        features.append(float(np.mean(bit0[1:, :] != bit0[:-1, :])))

    return np.asarray(features, dtype=np.float32)


def featurize(samples: list[tuple[Path, int]], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([extract_lsb_features(path, image_size=image_size) for path, _ in samples])
    y = np.asarray([label for _, label in samples], dtype=np.int64)
    return x, y


def train_lsb_logreg(
    train_samples: list[tuple[Path, int]],
    image_size: int,
) -> tuple[StandardScaler, LogisticRegression]:
    x_train, y_train = featurize(train_samples, image_size=image_size)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    return scaler, model


def evaluate_lsb_logreg(
    scaler: StandardScaler,
    model: LogisticRegression,
    test_samples: list[tuple[Path, int]],
    image_size: int,
) -> Metrics:
    x_test, y_test = featurize(test_samples, image_size=image_size)
    preds = model.predict(scaler.transform(x_test))
    pred_list = preds.tolist()
    label_list = y_test.tolist()

    return Metrics(
        accuracy=accuracy_score(label_list, pred_list),
        precision=precision_score(label_list, pred_list, zero_division=0),
        recall=recall_score(label_list, pred_list, zero_division=0),
        f1=f1_score(label_list, pred_list, zero_division=0),
        confusion_matrix=confusion_matrix(label_list, pred_list, labels=[0, 1]).tolist(),
        predicted_clean=pred_list.count(0),
        predicted_stego=pred_list.count(1),
    )
