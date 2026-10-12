from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.steganalysis.srnet import ManifestPairDataset, paired_batch


def write_pair(directory: Path, image_id: str) -> dict[str, object]:
    """One cover, and a stego that differs in exactly one pixel.

    The single changed pixel is the probe. Whatever rotation and flip the
    dataset applies, a correctly paired item moves that pixel the same way in
    both images, so exactly one pixel still differs.
    """
    rng = np.random.default_rng(abs(hash(image_id)) % (2**32))
    cover = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    stego = cover.copy()
    stego[7, 19] = (int(cover[7, 19]) + 40) % 256
    clean_path = directory / f"{image_id}_clean.png"
    stego_path = directory / f"{image_id}_stego.png"
    Image.fromarray(cover, mode="L").save(clean_path)
    Image.fromarray(stego, mode="L").save(stego_path)
    return {
        "image_id": image_id,
        "domain": "real",
        "split": "train",
        "clean_path": str(clean_path),
        "stego_path": str(stego_path),
    }


class PairedDatasetTests(unittest.TestCase):
    def build(self, directory: Path, count: int = 4, **kwargs) -> ManifestPairDataset:
        records = [write_pair(directory, f"img{index}") for index in range(count)]
        return ManifestPairDataset(
            records, "real", "train", preselected=True, **kwargs
        )

    def test_paired_length_counts_pairs_not_images(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.assertEqual(len(self.build(directory, paired=True)), 4)
            self.assertEqual(len(self.build(directory, paired=False)), 8)

    def test_paired_item_returns_both_images(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = self.build(Path(raw), paired=True)
            clean, stego, image_id = dataset[0]
            self.assertEqual(clean.shape, (1, 256, 256))
            self.assertEqual(stego.shape, (1, 256, 256))
            self.assertEqual(image_id, "img0")

    def test_augmentation_draws_once_for_the_pair(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = self.build(Path(raw), paired=True, augment=True)
            for attempt in range(40):
                random.seed(attempt)
                clean, stego, _ = dataset[attempt % len(dataset)]
                differing = int(torch.count_nonzero(clean - stego))
                self.assertEqual(
                    differing, 1,
                    f"attempt {attempt}: the pair received different transforms",
                )

    def test_flat_mode_keeps_label_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = self.build(Path(raw), paired=False)
            labels = sorted(int(dataset[index][1]) for index in range(len(dataset)))
            self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])


class PairedBatchTests(unittest.TestCase):
    def test_covers_come_first_and_labels_match(self) -> None:
        clean = torch.zeros(3, 1, 4, 4)
        stego = torch.ones(3, 1, 4, 4)
        images, labels = paired_batch(clean, stego, torch.device("cpu"))
        self.assertEqual(images.shape, (6, 1, 4, 4))
        self.assertEqual(labels.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertTrue(torch.equal(images[:3], clean))
        self.assertTrue(torch.equal(images[3:], stego))

    def test_every_batch_is_class_balanced(self) -> None:
        clean = torch.zeros(5, 1, 4, 4)
        stego = torch.ones(5, 1, 4, 4)
        _, labels = paired_batch(clean, stego, torch.device("cpu"))
        self.assertEqual(int((labels == 0).sum()), int((labels == 1).sum()))


if __name__ == "__main__":
    unittest.main()
