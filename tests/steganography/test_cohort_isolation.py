from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.steganography.core import (
    benchmark_manifest,
    cohort_identifier,
    embedding_ledger,
)


def build_cohort(directory: Path, fill: int, n: int = 2) -> Path:
    """Two cohorts can share image_ids while holding different pixels."""
    records = []
    rng = np.random.default_rng(fill)
    for domain in ("real", "ai"):
        for index in range(n):
            image_id = f"{domain}_{index}"
            path = directory / "img" / domain / f"{image_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            array = rng.integers(0, 255, (256, 256), dtype=np.uint8)
            Image.fromarray(array, mode="L").save(path)
            records.append({
                "image_id": image_id, "domain": domain, "split": "train",
                "clean_path": str(path), "source": "fixture",
                "width": 256, "height": 256, "color_mode": "L",
            })
    manifest = directory / "clean.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return manifest


class CohortIsolationTests(unittest.TestCase):
    def test_identifier_differs_for_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            a = build_cohort(Path(raw) / "a", fill=1)
            b = build_cohort(Path(raw) / "b", fill=2)
            self.assertNotEqual(cohort_identifier(a), cohort_identifier(b))

    def test_identifier_is_stable_for_same_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            a = build_cohort(Path(raw) / "a", fill=1)
            self.assertEqual(cohort_identifier(a), cohort_identifier(a))

    def test_cache_is_not_shared_between_cohorts(self) -> None:
        """Identical image_ids with different pixels must not reuse each other."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            first = build_cohort(root / "a", fill=1)
            second = build_cohort(root / "b", fill=2)
            run_a = benchmark_manifest(
                manifest=first, algorithm="suniward", payload_bpp=0.4, seed=42,
                output=root / "runs", cache_dir=cache,
            )
            run_b = benchmark_manifest(
                manifest=second, algorithm="suniward", payload_bpp=0.4, seed=42,
                output=root / "runs", cache_dir=cache,
            )
            config_b = json.loads((run_b.parent / "config.json").read_text())
            self.assertEqual(config_b["reused_count"], 0)
            self.assertEqual(config_b["generated_count"], 4)
            # and the stego images must actually differ
            rows_a = {json.loads(l)["image_id"]: json.loads(l)["stego_path"]
                      for l in open(run_a)}
            rows_b = {json.loads(l)["image_id"]: json.loads(l)["stego_path"]
                      for l in open(run_b)}
            for image_id, path_a in rows_a.items():
                self.assertNotEqual(path_a, rows_b[image_id])

    def test_same_cohort_reruns_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            manifest = build_cohort(root / "a", fill=1)
            benchmark_manifest(manifest=manifest, algorithm="suniward", payload_bpp=0.4,
                          seed=42, output=root / "runs", cache_dir=cache)
            second = benchmark_manifest(manifest=manifest, algorithm="suniward",
                                   payload_bpp=0.4, seed=42,
                                   output=root / "runs", cache_dir=cache)
            config = json.loads((second.parent / "config.json").read_text())
            self.assertEqual(config["generated_count"], 0)
            self.assertEqual(config["reused_count"], 4)

    def test_config_records_cohort_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = build_cohort(root / "a", fill=1)
            run = benchmark_manifest(manifest=manifest, algorithm="hill", payload_bpp=0.2,
                                seed=7, output=root / "runs", cache_dir=root / "cache")
            config = json.loads((run.parent / "config.json").read_text())
            self.assertEqual(config["cohort_id"], cohort_identifier(manifest))
            self.assertIn("input_manifest_sha256", config)


class LedgerTests(unittest.TestCase):
    def test_ledger_lists_conditions_already_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = build_cohort(root / "a", fill=1)
            for algorithm in ("suniward", "hill"):
                benchmark_manifest(manifest=manifest, algorithm=algorithm, payload_bpp=0.4,
                              seed=42, output=root / "runs", cache_dir=root / "cache")
            entries = embedding_ledger(root / "runs")
            algorithms = sorted(e["algorithm"] for e in entries)
            self.assertEqual(algorithms, ["hill", "suniward"])
            self.assertTrue(all(e["cohort_id"] == cohort_identifier(manifest)
                                for e in entries))

    def test_ledger_can_filter_to_one_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = build_cohort(root / "a", fill=1)
            second = build_cohort(root / "b", fill=2)
            benchmark_manifest(manifest=first, algorithm="suniward", payload_bpp=0.4,
                          seed=42, output=root / "runs", cache_dir=root / "cache")
            benchmark_manifest(manifest=second, algorithm="hill", payload_bpp=0.4,
                          seed=42, output=root / "runs", cache_dir=root / "cache")
            entries = embedding_ledger(root / "runs", cohort_id=cohort_identifier(first))
            self.assertEqual([e["algorithm"] for e in entries], ["suniward"])

    def test_ledger_on_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(embedding_ledger(Path(raw) / "none"), [])


if __name__ == "__main__":
    unittest.main()
