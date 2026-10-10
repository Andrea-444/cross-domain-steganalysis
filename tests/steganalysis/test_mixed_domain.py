from __future__ import annotations

import collections
import unittest

from src.steganalysis.srnet import select_training_records


def records(real: int, ai: int, split: str = "train") -> list[dict[str, object]]:
    rows = []
    for domain, n in (("real", real), ("ai", ai)):
        for index in range(n):
            rows.append({
                "image_id": f"{domain}_{index:04d}", "domain": domain, "split": split,
                "clean_path": f"/{domain}_{index}.png",
                "stego_path": f"/{domain}_{index}_s.png",
            })
    return rows


class SelectTrainingRecordsTests(unittest.TestCase):
    def test_single_domain_returns_that_domain_only(self) -> None:
        chosen = select_training_records(records(100, 100), "real", "train", seed=1)
        self.assertEqual({r["domain"] for r in chosen}, {"real"})
        self.assertEqual(len(chosen), 100)

    def test_mixed_matches_the_single_domain_size(self) -> None:
        """A mixed model trained on the union would see twice the data."""
        rows = records(100, 100)
        single = select_training_records(rows, "real", "train", seed=1)
        mixed = select_training_records(rows, "mixed", "train", seed=1)
        self.assertEqual(len(mixed), len(single))

    def test_mixed_draws_half_from_each_domain(self) -> None:
        mixed = select_training_records(records(100, 100), "mixed", "train", seed=1)
        counts = collections.Counter(r["domain"] for r in mixed)
        self.assertEqual(counts["real"], 50)
        self.assertEqual(counts["ai"], 50)

    def test_mixed_size_follows_the_smaller_domain(self) -> None:
        mixed = select_training_records(records(80, 100), "mixed", "train", seed=1)
        counts = collections.Counter(r["domain"] for r in mixed)
        self.assertEqual(len(mixed), 80)
        self.assertEqual(counts["real"], 40)
        self.assertEqual(counts["ai"], 40)

    def test_mixed_is_deterministic_for_a_seed(self) -> None:
        rows = records(100, 100)
        first = [r["image_id"] for r in select_training_records(rows, "mixed", "train", seed=7)]
        second = [r["image_id"] for r in select_training_records(rows, "mixed", "train", seed=7)]
        self.assertEqual(first, second)

    def test_a_different_seed_selects_a_different_subset(self) -> None:
        rows = records(100, 100)
        first = {r["image_id"] for r in select_training_records(rows, "mixed", "train", seed=1)}
        second = {r["image_id"] for r in select_training_records(rows, "mixed", "train", seed=2)}
        self.assertNotEqual(first, second)

    def test_odd_target_does_not_exceed_the_single_domain_size(self) -> None:
        mixed = select_training_records(records(101, 101), "mixed", "train", seed=1)
        self.assertLessEqual(len(mixed), 101)

    def test_unknown_domain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_training_records(records(10, 10), "nonexistent", "train", seed=1)

    def test_split_is_respected(self) -> None:
        rows = records(10, 10, split="train") + records(4, 4, split="val")
        chosen = select_training_records(rows, "mixed", "val", seed=1)
        self.assertTrue(all(r["split"] == "val" for r in chosen))
        self.assertEqual(len(chosen), 4)


if __name__ == "__main__":
    unittest.main()
