from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.manifest import write_manifest
from src.steganalysis.diagnostics import (
    add_arguments,
    check_embedding,
    check_provenance,
    classical_features,
    report_path,
    run,
    synthetic_stego,
)


def build_cohort(directory: Path, count: int, changed: float, seed: int = 0) -> Path:
    """A manifest of cover/stego pairs with a known change rate."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        cover = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        stego = synthetic_stego(cover, rng, changed)
        clean_path = directory / f"img{index}_clean.png"
        stego_path = directory / f"img{index}_stego.png"
        Image.fromarray(cover, mode="L").save(clean_path)
        Image.fromarray(stego, mode="L").save(stego_path)
        records.append(
            {
                "image_id": f"img{index}",
                "domain": "real",
                "split": "train" if index % 4 else "test",
                "clean_path": str(clean_path),
                "stego_path": str(stego_path),
                "algorithm": "synthetic",
                "payload_bpp": 0.4,
                "embedding_seed": seed,
                "source": "fixture",
            }
        )
    path = directory / "manifest.jsonl"
    write_manifest(path, records)
    return path


class SyntheticStegoTests(unittest.TestCase):
    def test_changes_are_plus_or_minus_one(self) -> None:
        rng = np.random.default_rng(0)
        cover = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        stego = synthetic_stego(cover, rng, 0.5)
        difference = stego.astype(np.int16) - cover.astype(np.int16)
        self.assertEqual(sorted(set(np.abs(difference).ravel().tolist())), [0, 1])

    def test_rate_controls_how_many_pixels_change(self) -> None:
        rng = np.random.default_rng(0)
        cover = np.full((128, 128), 128, dtype=np.uint8)
        for rate in (0.1, 0.5, 0.9):
            stego = synthetic_stego(cover, rng, rate)
            changed = np.count_nonzero(stego != cover) / cover.size
            self.assertAlmostEqual(changed, rate, delta=0.03)

    def test_values_stay_inside_the_byte_range(self) -> None:
        rng = np.random.default_rng(0)
        for value in (0, 255):
            cover = np.full((32, 32), value, dtype=np.uint8)
            stego = synthetic_stego(cover, rng, 1.0)
            self.assertGreaterEqual(int(stego.min()), 0)
            self.assertLessEqual(int(stego.max()), 255)


class EmbeddingCheckTests(unittest.TestCase):
    def test_reports_the_change_rate_it_was_given(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = build_cohort(Path(raw), count=8, changed=0.4)
            result = check_embedding(manifest, sample=8, seed=0)
            self.assertEqual(result["shared_paths"], 0)
            self.assertEqual(result["identical_pairs"], 0)
            self.assertAlmostEqual(result["changed_fraction_median"], 0.4, delta=0.05)
            self.assertEqual(result["absolute_difference_peaks"], [1])

    def test_identical_pairs_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = build_cohort(directory, count=4, changed=0.4)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            for record in records:
                record["stego_path"] = record["clean_path"]
            write_manifest(manifest, records)
            result = check_embedding(manifest, sample=4, seed=0)
            self.assertEqual(result["shared_paths"], 4)
            self.assertEqual(result["identical_pairs"], 4)


class ProvenanceCheckTests(unittest.TestCase):
    def test_walks_back_to_the_preprocessing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            prepared = directory / "cohort_clean.jsonl"
            prepared.write_text("", encoding="utf-8")
            (directory / "cohort_clean.config.json").write_text(
                json.dumps({"name": "cohort", "preprocess_mode": "crop", "size": 256}),
                encoding="utf-8",
            )
            balanced = directory / "cohort-balanced_clean.jsonl"
            balanced.write_text("", encoding="utf-8")
            (directory / "cohort-balanced_clean.config.json").write_text(
                json.dumps(
                    {
                        "name": "cohort-balanced",
                        "strategy": "min-count-per-class-and-split",
                        "source_manifest": str(prepared),
                    }
                ),
                encoding="utf-8",
            )
            run_config = directory / "config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "algorithm": "suniward",
                        "payload_bpp": 0.4,
                        "record_count": 10,
                        "input_manifest": str(balanced),
                    }
                ),
                encoding="utf-8",
            )
            result = check_provenance(run_config)
            self.assertEqual(result["preprocess_mode"], "crop")
            self.assertEqual(len(result["chain"]), 2)
            self.assertEqual(result["embedding"]["algorithm"], "suniward")

    def test_missing_config_ends_the_chain_without_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            orphan = directory / "unknown_clean.jsonl"
            orphan.write_text("", encoding="utf-8")
            run_config = directory / "config.json"
            run_config.write_text(
                json.dumps({"algorithm": "suniward", "input_manifest": str(orphan)}),
                encoding="utf-8",
            )
            result = check_provenance(run_config)
            self.assertIsNone(result["preprocess_mode"])
            self.assertFalse(result["chain"][0]["config_found"])


class ClassicalFeatureTests(unittest.TestCase):
    def test_descriptor_has_five_hundred_normalized_features(self) -> None:
        rng = np.random.default_rng(0)
        features = classical_features(rng.integers(0, 256, size=(64, 64), dtype=np.uint8))
        self.assertEqual(features.shape, (500,))
        for block in np.split(features, 4):
            self.assertAlmostEqual(float(block.sum()), 1.0, places=6)

    @staticmethod
    def smooth_cover(side: int = 128) -> np.ndarray:
        """A smooth image, as a photograph is. Uniform noise saturates the
        residuals, and an embedding then barely moves the descriptor."""
        grid = np.linspace(0, 1, side)
        rows, columns = np.meshgrid(grid, grid, indexing="ij")
        values = 128 + 100 * np.sin(6 * columns) * np.cos(4 * rows)
        return np.clip(values, 0, 255).astype(np.uint8)

    def test_descriptor_moves_further_as_the_embedding_grows(self) -> None:
        """The instrument must register a signal it is meant to catch.

        An absolute threshold would be arbitrary, so the test asserts the
        order instead: a heavier embedding must move the descriptor further.
        """
        cover = self.smooth_cover()
        distances = []
        for rate in (0.05, 0.25, 0.9):
            rng = np.random.default_rng(0)
            stego = synthetic_stego(cover, rng, rate)
            distances.append(
                float(np.abs(classical_features(cover) - classical_features(stego)).sum())
            )
        self.assertEqual(distances, sorted(distances))
        self.assertGreater(distances[-1], 5 * distances[0])


class ReportPathTests(unittest.TestCase):
    def test_none_means_no_report(self) -> None:
        self.assertIsNone(report_path(None, "embedding"))

    def test_a_directory_gets_a_timestamped_name(self) -> None:
        path = report_path("results/diagnostics/", "classical")
        self.assertEqual(path.parent, Path("results/diagnostics"))
        self.assertTrue(path.name.startswith("classical_"))
        self.assertTrue(path.name.endswith(".json"))

    def test_an_explicit_file_name_is_kept(self) -> None:
        self.assertEqual(report_path("out/one.json", "embedding"), Path("out/one.json"))


class CommandLineTests(unittest.TestCase):
    def parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_arguments(parser.add_subparsers(dest="check", required=True))
        return parser.parse_args(argv)

    def test_embedding_check_runs_and_writes_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = build_cohort(directory, count=6, changed=0.3)
            out = directory / "report.json"
            args = self.parse(
                ["embedding", "--manifest", str(manifest), "--sample", "6",
                 "--output", str(out)]
            )
            result = run(args)
            self.assertEqual(result["check"], "embedding")
            self.assertTrue(out.exists())
            self.assertEqual(json.loads(out.read_text())["sampled"], 6)

    def test_classical_check_defaults_are_declared(self) -> None:
        args = self.parse(["classical", "--manifest", "m.jsonl"])
        self.assertEqual(args.domain, "real")
        self.assertEqual(args.train_pairs, 1200)
        self.assertFalse(args.synthetic)

    def test_collapse_check_accepts_several_learning_rates(self) -> None:
        args = self.parse(
            ["collapse", "--manifest", "m.jsonl", "--learning-rates", "1e-3", "1e-4"]
        )
        self.assertEqual(args.learning_rates, [1e-3, 1e-4])

    def test_synthetic_train_defaults_to_the_lower_learning_rate(self) -> None:
        args = self.parse(["synthetic-train", "--manifest", "m.jsonl"])
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.synthetic_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
