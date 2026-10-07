from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.rebalance import rebalance_cohort


def manifest(directory: Path, counts: dict[tuple[str, str, str], int]) -> Path:
    """counts maps (class_id, split, domain) -> number of covers."""
    rows = []
    for (class_id, split, domain), n in counts.items():
        for index in range(n):
            rows.append({
                "image_id": f"{class_id}_{split}_{domain}_{index}",
                "class_id": class_id, "split": split, "domain": domain,
                "clean_path": f"/covers/{class_id}_{split}_{domain}_{index}.png",
                "width": 256, "height": 256, "color_mode": "L",
            })
    path = directory / "clean.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


class RebalanceTests(unittest.TestCase):
    def test_balanced_cohort_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "real"): 4, ("n1", "train", "ai"): 4})
            out = rebalance_cohort(src, name="bal", output=d / "out")
            self.assertEqual(len(load(out)), 8)

    def test_surplus_domain_is_trimmed_to_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "real"): 3, ("n1", "train", "ai"): 6})
            out = rebalance_cohort(src, name="bal", output=d / "out")
            counts = collections.Counter(r["domain"] for r in load(out))
            self.assertEqual(counts["real"], 3)
            self.assertEqual(counts["ai"], 3)

    def test_matching_is_per_class_and_split(self) -> None:
        """Covers are never moved between splits, so matching must respect them."""
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {
                ("n1", "train", "real"): 2, ("n1", "train", "ai"): 4,
                ("n1", "test", "real"): 1, ("n1", "test", "ai"): 1,
                ("n2", "train", "real"): 5, ("n2", "train", "ai"): 5,
            })
            rows = load(rebalance_cohort(src, name="bal", output=d / "out"))
            got = collections.Counter((r["class_id"], r["split"], r["domain"]) for r in rows)
            self.assertEqual(got[("n1", "train", "real")], 2)
            self.assertEqual(got[("n1", "train", "ai")], 2)
            self.assertEqual(got[("n1", "test", "ai")], 1)
            self.assertEqual(got[("n2", "train", "ai")], 5)

    def test_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "real"): 2, ("n1", "train", "ai"): 6})
            first = [r["image_id"] for r in load(rebalance_cohort(src, "a", d / "a"))]
            second = [r["image_id"] for r in load(rebalance_cohort(src, "b", d / "b"))]
            self.assertEqual(first, second)

    def test_records_what_was_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "real"): 1, ("n1", "train", "ai"): 4})
            out = rebalance_cohort(src, name="bal", output=d / "out")
            config = json.loads((out.parent / f"{out.stem}.config.json").read_text())
            self.assertEqual(config["dropped_count"], 3)
            self.assertEqual(config["dropped_by_domain"], {"ai": 3})
            self.assertEqual(config["source_record_count"], 5)
            self.assertEqual(config["record_count"], 2)

    def test_class_absent_from_one_domain_is_removed_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "ai"): 6, ("n2", "train", "real"): 2,
                               ("n2", "train", "ai"): 2})
            rows = load(rebalance_cohort(src, name="bal", output=d / "out"))
            self.assertEqual({r["class_id"] for r in rows}, {"n2"})

    def test_image_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            src = manifest(d, {("n1", "train", "real"): 2, ("n1", "train", "ai"): 3})
            before = {r["image_id"]: r["clean_path"] for r in load(src)}
            for row in load(rebalance_cohort(src, name="bal", output=d / "out")):
                self.assertEqual(row["clean_path"], before[row["image_id"]])


if __name__ == "__main__":
    unittest.main()
