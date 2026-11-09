from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.steganalysis.srnet import predict_srnet


def cohort(directory: Path, count: int = 4) -> list[dict[str, object]]:
    rng = np.random.default_rng(0)
    records = []
    for index in range(count):
        cover = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
        stego = cover.copy()
        stego[0, 0] = (int(cover[0, 0]) + 1) % 256
        clean_path = directory / f"img{index}_clean.png"
        stego_path = directory / f"img{index}_stego.png"
        Image.fromarray(cover, mode="L").save(clean_path)
        Image.fromarray(stego, mode="L").save(stego_path)
        records.append({
            "image_id": f"img{index}", "domain": "real", "split": "test",
            "clean_path": str(clean_path), "stego_path": str(stego_path),
        })
    return records


class RecordingModel(torch.nn.Module):
    """A tiny stand-in that reports whether autograd was on during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(256 * 256, 2)
        self.grad_enabled: list[bool] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        flat = images.view(images.size(0), -1)
        return torch.nn.functional.log_softmax(self.linear(flat), dim=1)


class EvaluationMemoryTests(unittest.TestCase):
    """Evaluation must not build an autograd graph.

    `eval()` only changes the normalization layers. Without `no_grad` the
    forward pass keeps the activations of every layer, so an evaluation costs
    about as much GPU memory as a training step. That memory is held while the
    trained model and the optimizer state are still on the card, and it caused
    an out-of-memory failure after training had finished.
    """

    def run_prediction(self) -> RecordingModel:
        model = RecordingModel()
        with tempfile.TemporaryDirectory() as raw:
            records = cohort(Path(raw))
            predict_srnet(
                model=model, records=records, test_domain="real",
                batch_size=4, device=torch.device("cpu"),
            )
        return model

    def test_autograd_is_off_during_every_forward_pass(self) -> None:
        model = self.run_prediction()
        self.assertTrue(model.grad_enabled)
        self.assertNotIn(True, model.grad_enabled)

    def test_the_returned_values_are_plain_numbers(self) -> None:
        model = RecordingModel()
        with tempfile.TemporaryDirectory() as raw:
            records = cohort(Path(raw))
            labels, predictions, probabilities, ids = predict_srnet(
                model=model, records=records, test_domain="real",
                batch_size=4, device=torch.device("cpu"),
            )
        self.assertEqual(len(labels), 8)
        self.assertEqual(sorted(set(labels)), [0, 1])
        for value in probabilities:
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_the_caller_keeps_its_own_grad_setting(self) -> None:
        """`no_grad` must not leak out of the function."""
        self.assertTrue(torch.is_grad_enabled())
        self.run_prediction()
        self.assertTrue(torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
