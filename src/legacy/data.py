from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.legacy.stego_lsb import IMAGE_EXTENSIONS


class CleanStegoDataset(Dataset):
    """Binary image dataset: clean=0, stego=1."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        input_mode: str = "rgb",
        clean_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.input_mode = input_mode
        if input_mode not in {"rgb", "residual"}:
            raise ValueError("input_mode must be 'rgb' or 'residual'.")
        self.samples: list[tuple[Path, int]] = []

        roots = {
            "clean": Path(clean_root) if clean_root is not None else self.root / "clean",
            "stego": self.root / "stego",
        }
        for class_name, label in (("clean", 0), ("stego", 1)):
            class_dir = roots[class_name]
            if not class_dir.exists():
                raise ValueError(f"Missing folder: {class_dir}")
            paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend((path, label) for path in paths)

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.input_mode == "residual":
            image = high_pass_residual(image)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        target = torch.tensor(label, dtype=torch.long)
        return image, target


def high_pass_residual(image: np.ndarray) -> np.ndarray:
    """Simple residual image that emphasizes local pixel artifacts."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    residual = image.astype(np.int16) - blurred.astype(np.int16)
    residual = np.clip(residual + 128, 0, 255).astype(np.uint8)
    return residual
