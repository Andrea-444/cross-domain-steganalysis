from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.datasets.preparation import UndersizedImageError, normalize_image


def gradient(width: int, height: int) -> Image.Image:
    array = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    return Image.fromarray(array, mode="L")


class CropOnlyTests(unittest.TestCase):
    def test_crop_mode_does_not_resample(self) -> None:
        """A native crop must reproduce source pixels exactly."""
        source = gradient(512, 512)
        result = normalize_image(source, size=256, mode="crop")
        self.assertEqual(result.size, (256, 256))
        expected = np.array(source)[128:384, 128:384]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_fit_mode_resamples(self) -> None:
        result = normalize_image(gradient(512, 512), size=256, mode="fit")
        self.assertEqual(result.size, (256, 256))

    def test_crop_mode_rejects_undersized_image(self) -> None:
        with self.assertRaises(UndersizedImageError):
            normalize_image(gradient(170, 300), size=256, mode="crop")

    def test_fit_mode_accepts_undersized_image(self) -> None:
        result = normalize_image(gradient(170, 300), size=256, mode="fit")
        self.assertEqual(result.size, (256, 256))

    def test_crop_is_centred(self) -> None:
        source = gradient(400, 300)
        result = np.array(normalize_image(source, size=256, mode="crop"))
        expected = np.array(source)[22:278, 72:328]
        np.testing.assert_array_equal(result, expected)

    def test_output_is_grayscale(self) -> None:
        colour = Image.new("RGB", (512, 512), (10, 200, 30))
        self.assertEqual(normalize_image(colour, size=256, mode="crop").mode, "L")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image(gradient(512, 512), size=256, mode="stretch")


if __name__ == "__main__":
    unittest.main()
