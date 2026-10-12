from __future__ import annotations

import math
import unittest

from src.evaluation.metrics import (
    calculate_metrics,
    equal_error_rate,
    generalization_gaps,
    summarize_directions,
)


class ThresholdMetricTests(unittest.TestCase):
    def metrics(self):
        # TN=3 FP=1 FN=2 TP=4
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        predictions = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
        return calculate_metrics(labels, predictions)

    def test_specificity_is_tn_over_actual_negatives(self) -> None:
        self.assertAlmostEqual(self.metrics()["specificity"], 3 / 4)

    def test_false_positive_rate(self) -> None:
        self.assertAlmostEqual(self.metrics()["false_positive_rate"], 1 / 4)

    def test_false_negative_rate(self) -> None:
        self.assertAlmostEqual(self.metrics()["false_negative_rate"], 2 / 6)

    def test_specificity_and_fpr_are_complements(self) -> None:
        m = self.metrics()
        self.assertAlmostEqual(m["specificity"] + m["false_positive_rate"], 1.0)

    def test_detection_error_is_the_mean_of_the_two_rates(self) -> None:
        m = self.metrics()
        self.assertAlmostEqual(
            m["detection_error"],
            0.5 * (m["false_positive_rate"] + m["false_negative_rate"]),
        )

    def test_perfect_prediction_gives_zero_detection_error(self) -> None:
        m = calculate_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(m["detection_error"], 0.0)
        self.assertAlmostEqual(m["mcc"], 1.0)

    def test_inverted_prediction_gives_detection_error_one(self) -> None:
        m = calculate_metrics([0, 0, 1, 1], [1, 1, 0, 0])
        self.assertAlmostEqual(m["detection_error"], 1.0)
        self.assertAlmostEqual(m["mcc"], -1.0)

    def test_mcc_is_zero_for_a_constant_prediction(self) -> None:
        self.assertAlmostEqual(calculate_metrics([0, 0, 1, 1], [1, 1, 1, 1])["mcc"], 0.0)


class ZeroDenominatorTests(unittest.TestCase):
    def test_no_actual_negatives_leaves_specificity_and_fpr_undefined(self) -> None:
        m = calculate_metrics([1, 1, 1], [1, 0, 1])
        self.assertIsNone(m["specificity"])
        self.assertIsNone(m["false_positive_rate"])
        self.assertIsNotNone(m["false_negative_rate"])

    def test_no_actual_positives_leaves_fnr_undefined(self) -> None:
        m = calculate_metrics([0, 0, 0], [0, 1, 0])
        self.assertIsNone(m["false_negative_rate"])
        self.assertIsNotNone(m["specificity"])

    def test_detection_error_is_undefined_when_either_rate_is(self) -> None:
        self.assertIsNone(calculate_metrics([1, 1, 1], [1, 0, 1])["detection_error"])

    def test_empty_input_is_safe(self) -> None:
        m = calculate_metrics([], [])
        self.assertIsNone(m["specificity"])
        self.assertIsNone(m["detection_error"])


class EqualErrorRateTests(unittest.TestCase):
    def test_separable_scores_give_an_eer_near_zero(self) -> None:
        labels = [0] * 50 + [1] * 50
        scores = [0.1] * 50 + [0.9] * 50
        self.assertLess(equal_error_rate(labels, scores), 0.01)

    def test_identical_scores_give_an_eer_near_one_half(self) -> None:
        labels = [0] * 50 + [1] * 50
        self.assertAlmostEqual(equal_error_rate(labels, [0.5] * 100), 0.5, places=2)

    def test_reversed_scores_give_a_high_eer(self) -> None:
        labels = [0] * 50 + [1] * 50
        scores = [0.9] * 50 + [0.1] * 50
        self.assertGreater(equal_error_rate(labels, scores), 0.99)

    def test_single_class_is_undefined(self) -> None:
        self.assertIsNone(equal_error_rate([1, 1, 1], [0.2, 0.7, 0.9]))

    def test_empty_input_is_undefined(self) -> None:
        self.assertIsNone(equal_error_rate([], []))

    def test_eer_appears_in_metrics_only_with_probabilities(self) -> None:
        labels = [0, 0, 1, 1]
        predictions = [0, 0, 1, 1]
        self.assertNotIn("eer", calculate_metrics(labels, predictions))
        with_scores = calculate_metrics(labels, predictions, [0.1, 0.2, 0.8, 0.9])
        self.assertIsNotNone(with_scores["eer"])

    def test_eer_is_none_for_single_class_labels(self) -> None:
        self.assertIsNone(calculate_metrics([1, 1], [1, 1], [0.8, 0.9])["eer"])


class GeneralizationGapTests(unittest.TestCase):
    def runs(self, values):
        """values maps direction -> list of balanced accuracies, one per seed."""
        out = []
        for index in range(len(next(iter(values.values())))):
            out.append({
                "seed": index + 1,
                "directions": {
                    direction: {
                        "balanced_accuracy": series[index], "accuracy": series[index],
                        "precision": series[index], "recall": series[index],
                        "f1": series[index],
                        "confusion_matrix": [[1, 0], [0, 1]],
                    }
                    for direction, series in values.items()
                },
            })
        return out

    def test_gap_is_paired_within_each_seed(self) -> None:
        runs = self.runs({
            "real_to_real": [0.90, 0.80], "real_to_ai": [0.60, 0.70],
            "ai_to_ai": [0.95, 0.85], "ai_to_real": [0.55, 0.65],
        })
        gaps = generalization_gaps(runs)
        real = gaps["real_to_real_minus_real_to_ai"]
        for got, want in zip(real["balanced_accuracy_per_seed"], [0.30, 0.10]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(real["balanced_accuracy_mean"], 0.20)

    def test_standard_deviation_is_reported(self) -> None:
        runs = self.runs({
            "real_to_real": [0.90, 0.80], "real_to_ai": [0.60, 0.70],
            "ai_to_ai": [0.95, 0.85], "ai_to_real": [0.55, 0.65],
        })
        self.assertAlmostEqual(
            gapstd := gaps_std(generalization_gaps(runs)), 0.10
        )

    def test_both_pairs_are_reported(self) -> None:
        runs = self.runs({
            "real_to_real": [0.9], "real_to_ai": [0.6],
            "ai_to_ai": [0.95], "ai_to_real": [0.55],
        })
        gaps = generalization_gaps(runs)
        self.assertIn("real_to_real_minus_real_to_ai", gaps)
        self.assertIn("ai_to_ai_minus_ai_to_real", gaps)
        self.assertAlmostEqual(
            gaps["ai_to_ai_minus_ai_to_real"]["balanced_accuracy_mean"], 0.40
        )

    def test_missing_direction_is_skipped(self) -> None:
        runs = self.runs({"real_to_real": [0.9], "real_to_ai": [0.6]})
        gaps = generalization_gaps(runs)
        self.assertIn("real_to_real_minus_real_to_ai", gaps)
        self.assertNotIn("ai_to_ai_minus_ai_to_real", gaps)

    def test_none_metric_is_skipped_without_failing(self) -> None:
        runs = [{
            "seed": 1,
            "directions": {
                "real_to_real": {"balanced_accuracy": 0.9, "eer": None},
                "real_to_ai": {"balanced_accuracy": 0.6, "eer": 0.3},
            },
        }]
        gaps = generalization_gaps(runs)["real_to_real_minus_real_to_ai"]
        self.assertIn("balanced_accuracy_mean", gaps)
        self.assertNotIn("eer_mean", gaps)

    def test_gaps_appear_in_the_summary(self) -> None:
        runs = self.runs({
            "real_to_real": [0.9], "real_to_ai": [0.6],
            "ai_to_ai": [0.95], "ai_to_real": [0.55],
        })
        summary = summarize_directions(runs, list(runs[0]["directions"]))
        self.assertIn("generalization_gaps", summary)


def gaps_std(gaps):
    return gaps["real_to_real_minus_real_to_ai"]["balanced_accuracy_std"]


if __name__ == "__main__":
    unittest.main()
