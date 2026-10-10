from __future__ import annotations

import unittest

from src.tracking.wandb import EVALUATION_COLUMNS, _flatten_summary, evaluation_rows


class EvaluationTableTests(unittest.TestCase):
    def runs(self):
        return [{
            "seed": 1,
            "directions": {
                "real_to_real": {
                    "accuracy": 0.9, "balanced_accuracy": 0.9, "precision": 0.9,
                    "recall": 0.9, "f1": 0.9, "roc_auc": 0.95, "pr_auc": 0.94,
                    "specificity": 0.88, "false_positive_rate": 0.12,
                    "false_negative_rate": 0.08, "detection_error": 0.10,
                    "mcc": 0.8, "eer": 0.11,
                    "confusion_matrix": [[44, 6], [4, 46]],
                },
            },
        }]

    def test_every_new_metric_has_a_column(self) -> None:
        for name in ("specificity", "false_positive_rate", "false_negative_rate",
                     "detection_error", "mcc", "eer"):
            self.assertIn(name, EVALUATION_COLUMNS)

    def test_a_row_matches_the_column_count(self) -> None:
        rows = evaluation_rows(self.runs())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(EVALUATION_COLUMNS))

    def test_row_values_line_up_with_their_columns(self) -> None:
        row = dict(zip(EVALUATION_COLUMNS, evaluation_rows(self.runs())[0]))
        self.assertEqual(row["seed"], 1)
        self.assertEqual(row["direction"], "real_to_real")
        self.assertAlmostEqual(row["detection_error"], 0.10)
        self.assertAlmostEqual(row["eer"], 0.11)
        self.assertEqual(row["true_clean"], 44)
        self.assertEqual(row["true_stego"], 46)

    def test_a_missing_metric_becomes_none_rather_than_shifting_columns(self) -> None:
        runs = self.runs()
        del runs[0]["directions"]["real_to_real"]["eer"]
        row = dict(zip(EVALUATION_COLUMNS, evaluation_rows(runs)[0]))
        self.assertIsNone(row["eer"])
        self.assertAlmostEqual(row["detection_error"], 0.10)


class FlattenSummaryTests(unittest.TestCase):
    def test_direction_metrics_are_flattened(self) -> None:
        flat = _flatten_summary({"real_to_real": {"balanced_accuracy_mean": 0.9}})
        self.assertAlmostEqual(flat["metrics/real_to_real/balanced_accuracy_mean"], 0.9)

    def test_generalization_gaps_are_flattened_under_their_own_prefix(self) -> None:
        summary = {
            "real_to_real": {"balanced_accuracy_mean": 0.9},
            "generalization_gaps": {
                "real_to_real_minus_real_to_ai": {
                    "balanced_accuracy_mean": 0.3,
                    "balanced_accuracy_std": 0.05,
                    "balanced_accuracy_per_seed": [0.25, 0.35],
                    "same_domain": "real_to_real",
                    "seeds": 2,
                },
            },
        }
        flat = _flatten_summary(summary)
        key = "gaps/real_to_real_minus_real_to_ai/balanced_accuracy_mean"
        self.assertAlmostEqual(flat[key], 0.3)
        self.assertAlmostEqual(
            flat["gaps/real_to_real_minus_real_to_ai/balanced_accuracy_std"], 0.05
        )

    def test_lists_and_strings_are_not_logged_as_metrics(self) -> None:
        flat = _flatten_summary({
            "generalization_gaps": {
                "real_to_real_minus_real_to_ai": {
                    "balanced_accuracy_per_seed": [0.25, 0.35],
                    "same_domain": "real_to_real",
                },
            },
        })
        self.assertEqual(flat, {})


if __name__ == "__main__":
    unittest.main()
