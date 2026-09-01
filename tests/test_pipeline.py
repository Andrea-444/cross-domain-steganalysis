from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.manifest import read_manifest, validate_clean_manifest
from src.datasets.pipeline import download_genimage, download_kaggle, prepare_domains
from src.steganalysis.experiments import (
    calculate_metrics,
    timestamped_results_dir,
    validate_pair_splits,
)
from src.steganalysis.srnet import DEFAULT_SRNET_ROOT, load_srnet_class
from src.steganography.core import (
    benchmark_manifest,
    embed_conseal,
    openstego_embed,
    openstego_extract,
    resolve_openstego_jar,
)


def write_image(path: Path, value: int, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value // 2, 255 - value)).save(path)


class DatasetTests(unittest.TestCase):
    def test_prepare_flattens_balances_and_splits_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "raw" / "real"
            ai = root / "raw" / "ai"
            for index in range(7):
                write_image(real / f"category_{index % 2}" / f"r{index}.jpg", 20 + index, (80, 60))
            for index in range(5):
                write_image(ai / f"folder_{index % 2}" / f"a{index}.png", 80 + index, (64, 64))

            manifest = prepare_domains(
                real_src=real,
                ai_src=ai,
                output=root / "processed",
                name="test",
                source="fixture",
                size=256,
                seed=9,
            )
            records = read_manifest(manifest)
            validate_clean_manifest(records)
            self.assertEqual(len(records), 10)
            self.assertEqual(sum(record["domain"] == "real" for record in records), 5)
            self.assertEqual(sum(record["domain"] == "ai" for record in records), 5)
            self.assertNotIn("category", records[0])
            for record in records:
                with Image.open(record["clean_path"]) as image:
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (256, 256))
            for domain in ("real", "ai"):
                splits = {
                    record["split"] for record in records if record["domain"] == domain
                }
                self.assertEqual(splits, {"train", "val", "test"})

    def test_download_validation_errors_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "owner/dataset-name"):
            download_kaggle("missing-slug", "/tmp/not-used")
        with self.assertRaisesRegex(RuntimeError, "accept-large-download"):
            download_genimage(
                "stable-diffusion-v1-4",
                "/tmp/not-used",
                accept_large_download=False,
            )


class SteganographyTests(unittest.TestCase):
    def test_conseal_is_deterministic_for_fixed_seed(self) -> None:
        rng = np.random.default_rng(3)
        cover = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        first = embed_conseal(cover, "suniward", 0.2, seed=123)
        second = embed_conseal(cover, "suniward", 0.2, seed=123)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(np.count_nonzero(first != cover), 0)
        self.assertLessEqual(
            int(np.abs(first.astype(np.int16) - cover.astype(np.int16)).max()),
            1,
        )

    def test_benchmark_reuses_cached_stego_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "raw" / "real"
            ai = root / "raw" / "ai"
            for index in range(3):
                write_image(real / f"r{index}.png", 20 + index, (80, 80))
                write_image(ai / f"a{index}.png", 90 + index, (80, 80))

            manifest = prepare_domains(
                real_src=real,
                ai_src=ai,
                output=root / "processed",
                name="cachetest",
                source="fixture",
                size=64,
                seed=7,
            )
            first_manifest = benchmark_manifest(
                manifest=manifest,
                algorithm="wow",
                payload_bpp=0.2,
                output=root / "runs",
                seed=42,
                backend="conseal",
                cache_dir=root / "cache",
                limit_per_domain=2,
            )
            second_manifest = benchmark_manifest(
                manifest=manifest,
                algorithm="wow",
                payload_bpp=0.2,
                output=root / "runs",
                seed=42,
                backend="conseal",
                cache_dir=root / "cache",
                limit_per_domain=2,
            )
            self.assertEqual(
                [record["stego_path"] for record in read_manifest(first_manifest)],
                [record["stego_path"] for record in read_manifest(second_manifest)],
            )

            configs = sorted((root / "runs").glob("*/config.json"))
            self.assertEqual(len(configs), 2)
            second_config = json.loads(configs[1].read_text(encoding="utf-8"))
            self.assertEqual(second_config["generated_count"], 0)
            self.assertEqual(second_config["reused_count"], 4)
            self.assertEqual(second_config["record_count"], 4)
            self.assertEqual(second_config["limit_per_domain"], 2)

    @unittest.skipUnless(
        Path("external/openstego/openstego.jar").exists(),
        "OpenStego is not installed",
    )
    def test_openstego_recovers_message_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cover = root / "cover.png"
            write_image(cover, 70, (256, 256))
            payloads = {
                "message": root / "secret.txt",
                "image": root / "secret.png",
            }
            payloads["message"].write_bytes(b"research payload")
            write_image(payloads["image"], 120, (8, 8))
            jar = resolve_openstego_jar()

            for payload_type, payload in payloads.items():
                stego = root / f"{payload_type}_stego.png"
                extracted = root / f"{payload_type}_out"
                openstego_embed(
                    payload_type=payload_type,
                    payload_file=payload,
                    cover=cover,
                    output=stego,
                    jar=jar,
                    max_bits=1,
                    password_env="TEST_OPENSTEGO_PASSWORD",
                )
                openstego_extract(
                    stego=stego,
                    output_dir=extracted,
                    jar=jar,
                    password_env="TEST_OPENSTEGO_PASSWORD",
                )
                self.assertEqual(payload.read_bytes(), (extracted / payload.name).read_bytes())


class ExperimentTests(unittest.TestCase):
    def test_metrics_and_pair_validation(self) -> None:
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        validate_pair_splits(
            [
                {"image_id": "a", "split": "train"},
                {"image_id": "a", "split": "train"},
            ]
        )
        with self.assertRaises(ValueError):
            validate_pair_splits(
                [
                    {"image_id": "a", "split": "train"},
                    {"image_id": "a", "split": "test"},
                ]
            )

    def test_timestamped_result_folder_contains_detector_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = timestamped_results_dir(temporary, "srnet")
            self.assertTrue(path.is_dir())
            self.assertTrue(path.name.startswith("srnet_"))

    @unittest.skipUnless(DEFAULT_SRNET_ROOT.exists(), "SRNet reference is not installed")
    def test_reference_srnet_forward_shape(self) -> None:
        model_class = load_srnet_class()
        model = model_class()
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros((1, 1, 256, 256), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
