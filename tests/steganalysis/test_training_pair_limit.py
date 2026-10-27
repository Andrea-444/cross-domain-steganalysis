from __future__ import annotations

import unittest

from src.steganalysis.srnet import select_training_records


def cohort(real: int = 100, ai: int = 100) -> list[dict[str, object]]:
    records = []
    for domain, count in (("real", real), ("ai", ai)):
        for index in range(count):
            records.append(
                {
                    "image_id": f"{domain}_{index:04d}",
                    "domain": domain,
                    "split": "train",
                    "clean_path": f"/{domain}_{index}.png",
                    "stego_path": f"/{domain}_{index}_stego.png",
                }
            )
    return records


class TrainingPairLimitTests(unittest.TestCase):
    """The limit answers how much of a result comes from the data quantity."""

    def test_no_limit_keeps_every_pair(self) -> None:
        kept = select_training_records(cohort(), "real", "train", seed=1)
        self.assertEqual(len(kept), 100)

    def test_the_limit_caps_the_count(self) -> None:
        kept = select_training_records(cohort(), "real", "train", seed=1, limit=40)
        self.assertEqual(len(kept), 40)

    def test_a_limit_above_the_cohort_keeps_every_pair(self) -> None:
        kept = select_training_records(cohort(), "real", "train", seed=1, limit=500)
        self.assertEqual(len(kept), 100)

    def test_the_same_seed_gives_the_same_subset(self) -> None:
        first = select_training_records(cohort(), "real", "train", seed=1, limit=40)
        second = select_training_records(cohort(), "real", "train", seed=1, limit=40)
        self.assertEqual([r["image_id"] for r in first], [r["image_id"] for r in second])

    def test_a_smaller_limit_sits_inside_a_larger_one(self) -> None:
        """A subset relation keeps two budgets comparable."""
        larger = select_training_records(cohort(), "real", "train", seed=1, limit=60)
        smaller = select_training_records(cohort(), "real", "train", seed=1, limit=30)
        self.assertTrue(
            {r["image_id"] for r in smaller} <= {r["image_id"] for r in larger}
        )

    def test_the_subset_is_not_simply_the_first_rows(self) -> None:
        """Taking the head of a sorted list would bias the choice."""
        kept = select_training_records(cohort(), "real", "train", seed=1, limit=20)
        head = [f"real_{index:04d}" for index in range(20)]
        self.assertNotEqual([str(r["image_id"]) for r in kept], head)

    def test_the_result_stays_sorted_by_image_id(self) -> None:
        kept = select_training_records(cohort(), "real", "train", seed=1, limit=25)
        identifiers = [str(r["image_id"]) for r in kept]
        self.assertEqual(identifiers, sorted(identifiers))

    def test_the_mixed_domain_respects_the_limit_and_stays_balanced(self) -> None:
        kept = select_training_records(cohort(), "mixed", "train", seed=1, limit=40)
        self.assertEqual(len(kept), 40)
        counts = {"real": 0, "ai": 0}
        for record in kept:
            counts[str(record["domain"])] += 1
        self.assertEqual(counts["real"], counts["ai"])

    def test_a_limit_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            select_training_records(cohort(), "real", "train", seed=1, limit=0)

    def test_the_limit_leaves_the_validation_split_alone(self) -> None:
        """Only the training count varies. A smaller validation set would
        change how the run chooses its checkpoint."""
        records = cohort()
        for record in records[:20]:
            record["split"] = "val"
        kept = select_training_records(records, "real", "val", seed=1)
        self.assertEqual(len(kept), 20)


if __name__ == "__main__":
    unittest.main()
