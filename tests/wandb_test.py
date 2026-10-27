from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tracking.wandb import start_experiment_run


class FakeRun:
    id = "run-123"
    url = "https://wandb.invalid/run-123"

    def __init__(self) -> None:
        self.logged = []
        self.metrics = []
        self.finished = False
        self.summary = FakeSummary()

    def define_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))

    def log(self, values) -> None:
        self.logged.append(values)

    def log_artifact(self, artifact) -> None:
        self.artifact = artifact

    def finish(self) -> None:
        self.finished = True


class FakeSummary(dict):
    def update(self, values) -> None:  # type: ignore[override]
        super().update(values)


class FakeTable:
    def __init__(self, columns) -> None:
        self.columns = columns
        self.rows = []

    def add_data(self, *values) -> None:
        self.rows.append(values)


class WandbExperimentTrackingTests(unittest.TestCase):
    def test_disabled_mode_does_not_import_wandb(self) -> None:
        tracker = start_experiment_run(
            mode="disabled", project="test", entity=None, name="test",
            config={}, tags=None, upload_artifact=False,
        )
        self.assertIsNone(tracker)

    def test_training_curves_go_to_history_and_metrics_go_to_summary(self) -> None:
        fake_run = FakeRun()
        fake_wandb = types.SimpleNamespace(
            init=lambda **kwargs: fake_run,
            Table=FakeTable,
        )
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = start_experiment_run(
                mode="online", project="test", entity=None, name="test",
                config={"epochs": 1}, tags=["smoke"], upload_artifact=False,
            )
            tracker.log_epoch(
                1, "real", {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6}
            )
            metrics = {
                "accuracy": 0.75,
                "balanced_accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.74,
                "roc_auc": 0.81,
                "pr_auc": 0.82,
                "confusion_matrix": [[8, 2], [3, 7]],
            }
            with tempfile.TemporaryDirectory() as raw:
                status = tracker.finish(
                    {
                        "runs": [{"seed": 1, "directions": {"real_to_ai": metrics}}],
                        "summary": {
                            "real_to_ai": {
                                "accuracy_mean": 0.75,
                                "accuracy_std": 0.0,
                            }
                        },
                    },
                    Path(raw),
                )

        self.assertEqual(status["status"], "logged")
        self.assertEqual(status["run_id"], "run-123")
        self.assertTrue(fake_run.finished)
        self.assertTrue(
            any("training/seed_1/real/train_loss" in item for item in fake_run.logged)
        )
        final_table = next(
            item["evaluation/results"]
            for item in fake_run.logged
            if "evaluation/results" in item
        )
        self.assertEqual(len(final_table.rows), 1)

        # The aggregate belongs in the summary, where it is a column of the
        # runs table rather than a panel with one bar.
        self.assertAlmostEqual(
            fake_run.summary["metrics/real_to_ai/accuracy_mean"], 0.75
        )

    def test_no_single_value_metric_reaches_the_history(self) -> None:
        """Every key in the history must be a training curve or the table.

        W&B draws a panel for each history key. A final metric logged there
        becomes a chart with one bar, and a protocol run produced hundreds.
        """
        fake_run = FakeRun()
        fake_wandb = types.SimpleNamespace(init=lambda **kwargs: fake_run, Table=FakeTable)
        metrics = {
            "accuracy": 0.75, "balanced_accuracy": 0.75, "roc_auc": 0.81,
            "confusion_matrix": [[8, 2], [3, 7]],
        }
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = start_experiment_run(
                mode="online", project="test", entity=None, name="test",
                config={}, tags=None, upload_artifact=False,
            )
            tracker.log_epoch(1, "real", {"epoch": 1, "train_loss": 0.7})
            with tempfile.TemporaryDirectory() as raw:
                tracker.finish(
                    {
                        "runs": [{"seed": 1, "directions": {"real_to_ai": metrics}}],
                        "summary": {"real_to_ai": {"accuracy_mean": 0.75}},
                    },
                    Path(raw),
                )
        keys = {key for item in fake_run.logged for key in item}
        for key in keys:
            self.assertTrue(
                key.startswith("training/") or key == "evaluation/results",
                f"{key} would become a panel",
            )
        self.assertNotIn("metrics/real_to_ai/accuracy_mean", keys)

    def test_the_tracker_no_longer_offers_per_seed_scalar_logging(self) -> None:
        """The table carries per-seed detail. Two homes would diverge."""
        from src.tracking.wandb import WandbExperimentTracker

        self.assertFalse(hasattr(WandbExperimentTracker, "log_evaluation"))


if __name__ == "__main__":
    unittest.main()
