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

    def define_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))

    def log(self, values) -> None:
        self.logged.append(values)

    def log_artifact(self, artifact) -> None:
        self.artifact = artifact

    def finish(self) -> None:
        self.finished = True


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

    def test_logs_epochs_evaluations_and_final_table(self) -> None:
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
            tracker.log_evaluation(1, "real_to_ai", metrics)
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


if __name__ == "__main__":
    unittest.main()
