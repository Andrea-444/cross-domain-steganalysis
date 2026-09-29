from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.recompress import recompress_cohort


def build_cohort(directory: Path, n: int = 3) -> Path:
    rng = np.random.default_rng(0)
    records = []
    for domain in ("real", "ai"):
        for index in range(n):
            path = directory / "src" / domain / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                rng.integers(0, 255, (256, 256), dtype=np.uint8), mode="L"
            ).save(path)
            records.append({
                "image_id": f"{domain}_{index}", "domain": domain, "split": "train",
                "clean_path": str(path), "class_id": "n0000000",
                "width": 256, "height": 256, "color_mode": "L",
            })
    manifest = directory / "clean.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return manifest


def load(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"))


class RecompressTests(unittest.TestCase):
    def test_only_named_domains_are_changed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = build_cohort(directory)
            out = recompress_cohort(
                manifest, name="variant", output=directory / "out",
                quality=90, domains=["ai"],
            )
            before = {json.loads(l)["image_id"]: json.loads(l)["clean_path"]
                      for l in open(manifest)}
            after = [json.loads(l) for l in open(out)]
            for record in after:
                original = load(before[record["image_id"]])
                produced = load(record["clean_path"])
                if record["domain"] == "ai":
                    self.assertFalse(np.array_equal(original, produced))
                else:
                    np.testing.assert_array_equal(original, produced)

    def test_output_stays_lossless_png_at_same_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            out = recompress_cohort(
                build_cohort(directory), name="variant",
                output=directory / "out", quality=90, domains=["ai", "real"],
            )
            for line in open(out):
                record = json.loads(line)
                with Image.open(record["clean_path"]) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.size, (256, 256))
                    self.assertEqual(image.mode, "L")

    def test_records_the_recompression_applied(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            out = recompress_cohort(
                build_cohort(directory), name="variant",
                output=directory / "out", quality=88, domains=["ai"],
            )
            records = [json.loads(l) for l in open(out)]
            ai = [r for r in records if r["domain"] == "ai"]
            real = [r for r in records if r["domain"] == "real"]
            self.assertTrue(all(r["jpeg_quality_applied"] == 88 for r in ai))
            self.assertTrue(all(r["jpeg_quality_applied"] is None for r in real))
            config = json.loads(
                (out.parent / f"{out.stem}.config.json").read_text()
            )
            self.assertEqual(config["quality"], 88)
            self.assertEqual(config["domains"], ["ai"])

    def test_lower_quality_changes_more(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = build_cohort(directory)
            deltas = {}
            for quality in (95, 40):
                out = recompress_cohort(
                    manifest, name=f"q{quality}", output=directory / f"o{quality}",
                    quality=quality, domains=["ai"],
                )
                before = {json.loads(l)["image_id"]: json.loads(l)["clean_path"]
                          for l in open(manifest)}
                diffs = [
                    float(np.abs(load(r["clean_path"]).astype(float)
                                 - load(before[r["image_id"]]).astype(float)).mean())
                    for r in map(json.loads, open(out)) if r["domain"] == "ai"
                ]
                deltas[quality] = sum(diffs) / len(diffs)
            self.assertGreater(deltas[40], deltas[95])

    def test_rejects_unknown_domain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaises(ValueError):
                recompress_cohort(
                    build_cohort(directory), name="v", output=directory / "out",
                    quality=90, domains=["nonexistent"],
                )


if __name__ == "__main__":
    unittest.main()
