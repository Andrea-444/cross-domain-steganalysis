from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.protocol import (
    CONDITION_FIELDS,
    DIRECTION_ORDER,
    aggregate,
    condition_key,
    escape_epoch,
    merge_by_seed,
    present_directions,
    read_run,
)


def metrics(balanced: float, auc: float) -> dict[str, object]:
    return {
        "accuracy": balanced, "balanced_accuracy": balanced, "precision": balanced,
        "recall": balanced, "f1": balanced, "specificity": balanced,
        "false_positive_rate": 1 - balanced, "false_negative_rate": 1 - balanced,
        "detection_error": 1 - balanced, "mcc": 2 * balanced - 1,
        "roc_auc": auc, "pr_auc": auc, "eer": 1 - auc,
        "confusion_matrix": [[10, 2], [3, 9]],
    }


def write_run(
    root: Path, folder: str, seed: int, domain: str,
    directions: dict[str, dict[str, object]], *,
    manifest: str = "abc123", learning_rate: float = 1e-4, epochs: int = 300,
    escape_at: int = 18, selected: int = 287,
    train_pairs: int | None = None,
    mixed_pairs_per_domain: int | None = None,
) -> Path:
    run = root / folder
    (run / "history").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "learning_rate": learning_rate, "epochs": epochs, "batch_size": 32,
        "train_pairs": train_pairs,
        "mixed_pairs_per_domain": mixed_pairs_per_domain,
        "condition": {"algorithm": "suniward", "payload_bpp": 0.4,
                      "preprocess_mode": "crop"},
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "runs": [{"seed": seed, "directions": directions}],
        "summary": {},
        "provenance": {
            "manifest_sha256": manifest, "detector_commit": "2c838cf7",
            "repository_commit": "d5168c07", "repository_dirty": False,
            "device": "cuda", "duration_seconds": 13320.0,
            "embedding": {"algorithm": "suniward", "payload_bpp": 0.4},
        },
    }), encoding="utf-8")
    epoch_list = [
        {"epoch": e, "train_loss": 0.6931 if e < escape_at else 0.5,
         "train_accuracy": 0.79, "val_balanced_accuracy": 0.75}
        for e in range(1, epochs + 1)
    ]
    (run / "history" / f"seed{seed}_{domain}.json").write_text(json.dumps({
        "seed": seed, "train_domain": domain, "epochs": epoch_list,
        "selection": {"criterion": "val_loss", "epoch": selected},
    }), encoding="utf-8")
    return run


class EscapeEpochTests(unittest.TestCase):
    def test_finds_the_first_epoch_below_the_plateau(self) -> None:
        epochs = [{"epoch": e, "train_loss": 0.6931} for e in range(1, 10)]
        epochs.append({"epoch": 10, "train_loss": 0.69})
        self.assertEqual(escape_epoch(epochs), 10)

    def test_a_detector_that_never_escapes_returns_none(self) -> None:
        self.assertIsNone(
            escape_epoch([{"epoch": e, "train_loss": 0.69315} for e in range(1, 50)])
        )


class MergeTests(unittest.TestCase):
    def runs(self):
        return [
            {"folder": "a", "runs": [
                {"seed": 1, "directions": {"real_to_real": metrics(0.76, 0.88),
                                           "real_to_ai": metrics(0.51, 0.53)}}]},
            {"folder": "b", "runs": [
                {"seed": 1, "directions": {"ai_to_ai": metrics(0.62, 0.70),
                                           "ai_to_real": metrics(0.51, 0.53)}}]},
            {"folder": "c", "runs": [
                {"seed": 2, "directions": {"real_to_real": metrics(0.76, 0.88),
                                           "real_to_ai": metrics(0.50, 0.53)}}]},
        ]

    def test_one_seed_gathers_directions_from_several_folders(self) -> None:
        merged, conflicts = merge_by_seed(self.runs())
        self.assertEqual([r["seed"] for r in merged], [1, 2])
        self.assertEqual(
            sorted(merged[0]["directions"]),
            ["ai_to_ai", "ai_to_real", "real_to_ai", "real_to_real"],
        )
        self.assertEqual(conflicts, [])

    def test_a_repeated_direction_is_reported_and_the_later_folder_wins(self) -> None:
        runs = self.runs()
        runs.append({"folder": "z", "runs": [
            {"seed": 1, "directions": {"real_to_real": metrics(0.80, 0.90)}}]})
        merged, conflicts = merge_by_seed(runs)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["kept"], "z")
        self.assertEqual(conflicts[0]["replaced"], "a")
        self.assertAlmostEqual(merged[0]["directions"]["real_to_real"]["roc_auc"], 0.90)

    def test_a_direction_missing_from_one_seed_is_left_out(self) -> None:
        """A mean over a different seed count than its neighbours misleads."""
        merged, _ = merge_by_seed(self.runs())
        kept = present_directions(merged)
        self.assertIn("real_to_real", kept)
        self.assertNotIn("ai_to_ai", kept)  # seed 2 has no AI run

    def test_kept_directions_follow_the_protocol_order(self) -> None:
        merged = [{"seed": 1, "directions": {
            "ai_to_real": {}, "real_to_real": {}, "mixed_to_ai": {}}}]
        self.assertEqual(
            present_directions(merged),
            ["real_to_real", "mixed_to_ai", "ai_to_real"],
        )


class DirectionOrderTests(unittest.TestCase):
    def test_the_order_matches_the_runner(self) -> None:
        """Two lists of direction names must not drift apart."""
        from src.steganalysis.experiments import DIRECTIONS

        self.assertEqual(list(DIRECTION_ORDER), list(DIRECTIONS))


class AggregateTests(unittest.TestCase):
    def build(self, root: Path) -> None:
        for seed in (1, 2, 3):
            write_run(root, f"srnet_2026101{seed}_real", seed, "real",
                      {"real_to_real": metrics(0.76 + seed / 1000, 0.88),
                       "real_to_ai": metrics(0.51, 0.53)})
            write_run(root, f"srnet_2026101{seed}_ai", seed, "ai",
                      {"ai_to_ai": metrics(0.62, 0.70),
                       "ai_to_real": metrics(0.51, 0.53)})

    def test_cross_seed_mean_and_spread_exist(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.build(root)
            result = aggregate(sorted(root.glob("srnet_*")))[0]
        self.assertEqual(result["seeds"], [1, 2, 3])
        values = result["summary"]["real_to_real"]
        self.assertAlmostEqual(values["balanced_accuracy_mean"], 0.762, places=6)
        self.assertGreater(values["balanced_accuracy_std"], 0.0)

    def test_paired_gaps_are_averaged_over_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.build(root)
            result = aggregate(sorted(root.glob("srnet_*")))[0]
        gaps = result["summary"]["generalization_gaps"]
        self.assertEqual(gaps["real_to_real_minus_real_to_ai"]["seeds"], 3)
        self.assertEqual(gaps["ai_to_ai_minus_ai_to_real"]["seeds"], 3)

    def test_a_different_condition_is_a_separate_group(self) -> None:
        """Averaging two learning rates would hide the difference."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "srnet_a", 1, "real", {"real_to_real": metrics(0.76, 0.88)})
            write_run(root, "srnet_b", 2, "real", {"real_to_real": metrics(0.50, 0.50)},
                      learning_rate=1e-3)
            results = aggregate(sorted(root.glob("srnet_*")))
        self.assertEqual(len(results), 2)
        rates = sorted(r["condition"]["learning_rate"] for r in results)
        self.assertEqual(rates, [1e-4, 1e-3])

    def test_a_different_mixed_training_size_is_a_separate_group(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "srnet_a", 1, "mixed", {"mixed_to_real": metrics(0.7, 0.8)})
            write_run(
                root,
                "srnet_b",
                1,
                "mixed",
                {"mixed_to_real": metrics(0.75, 0.85)},
                mixed_pairs_per_domain=3887,
            )
            results = aggregate(sorted(root.glob("srnet_*")))
        self.assertEqual(len(results), 2)

    def test_training_facts_are_carried_through(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "srnet_a", 1, "real", {"real_to_real": metrics(0.76, 0.88)},
                      escape_at=19, selected=287)
            result = aggregate(sorted(root.glob("srnet_*")))[0]
        entry = result["training"][0]
        self.assertEqual(entry["escape_epoch"], 19)
        self.assertEqual(entry["selected_epoch"], 287)
        self.assertEqual(entry["total_epochs"], 300)

    def test_an_unfinished_run_folder_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "srnet_a", 1, "real", {"real_to_real": metrics(0.76, 0.88)})
            (root / "srnet_broken").mkdir()
            self.assertIsNone(read_run(root / "srnet_broken"))
            results = aggregate(sorted(root.glob("srnet_*")))
        self.assertEqual(len(results), 1)

    def test_condition_key_covers_every_declared_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "srnet_a", 1, "real", {"real_to_real": metrics(0.7, 0.8)})
            run = read_run(root / "srnet_a")
        self.assertEqual(len(condition_key(run)), len(CONDITION_FIELDS))
        self.assertEqual(condition_key(run)[-2:], (None, None))


if __name__ == "__main__":
    unittest.main()
