from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.audit import (
    audit_domain_separability,
    blockiness_score,
    highpass_energy,
    image_features,
)


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def smooth_image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 4, (size, size))
    # low-pass it so high-frequency content is minimal
    for _ in range(3):
        base[1:-1, 1:-1] = (
            base[:-2, 1:-1] + base[2:, 1:-1] + base[1:-1, :-2] + base[1:-1, 2:]
        ) / 4
    return np.clip(base, 0, 255)


def blocky_image(seed: int, size: int = 64) -> np.ndarray:
    """Piecewise-constant 8x8 tiles: the signature JPEG leaves behind."""
    rng = np.random.default_rng(seed)
    tiles = rng.normal(128, 25, (size // 8, size // 8))
    return np.clip(np.kron(tiles, np.ones((8, 8))), 0, 255)


class FeatureTests(unittest.TestCase):
    def test_blockiness_higher_for_tiled_image(self) -> None:
        self.assertGreater(
            blockiness_score(blocky_image(1)), blockiness_score(smooth_image(1))
        )

    def test_highpass_energy_higher_for_noisy_image(self) -> None:
        rng = np.random.default_rng(0)
        noisy = np.clip(rng.normal(128, 30, (64, 64)), 0, 255)
        self.assertGreater(highpass_energy(noisy), highpass_energy(smooth_image(0)))

    def test_image_features_returns_finite_named_values(self) -> None:
        features = image_features(smooth_image(3))
        self.assertTrue(features)
        for name, value in features.items():
            self.assertTrue(np.isfinite(value), f"{name} is not finite")


class SeparabilityTests(unittest.TestCase):
    def build(self, directory: Path, distinguishable: bool) -> Path:
        records = []
        for index in range(20):
            for domain in ("real", "ai"):
                if distinguishable:
                    array = (
                        blocky_image(index) if domain == "real" else smooth_image(index)
                    )
                else:
                    array = smooth_image(index * 2 + (domain == "ai"))
                path = directory / domain / f"{domain}_{index}.png"
                write_image(path, array)
                records.append(
                    {"image_id": f"{domain}_{index}", "domain": domain,
                     "clean_path": str(path), "split": "train"}
                )
        manifest = directory / "manifest.jsonl"
        import json
        manifest.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        return manifest

    def test_detects_confounded_domains(self) -> None:
        """Blocky vs smooth covers must be flagged as trivially separable."""
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.build(Path(raw), distinguishable=True)
            result = audit_domain_separability(manifest, folds=4, seed=1)
            self.assertGreater(result["balanced_accuracy"], 0.9)
            self.assertEqual(result["counts"], {"real": 20, "ai": 20})

    def test_indistinguishable_domains_score_near_chance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.build(Path(raw), distinguishable=False)
            result = audit_domain_separability(manifest, folds=4, seed=1)
            self.assertLess(result["balanced_accuracy"], 0.8)

    def test_reports_per_domain_feature_means(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.build(Path(raw), distinguishable=True)
            result = audit_domain_separability(manifest, folds=4, seed=1)
            self.assertIn("blockiness", result["per_domain"]["real"])
            self.assertIn("roc_auc", result)


if __name__ == "__main__":
    unittest.main()
