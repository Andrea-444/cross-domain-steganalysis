from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tracking.naming import (
    format_learning_rate,
    format_payload,
    format_seeds,
    manifest_condition,
    run_group,
    run_name,
    run_role,
    run_tags,
)


CONDITION = {"algorithm": "suniward", "payload_bpp": 0.4, "preprocess_mode": "crop"}
SHA = "f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873"


class FormatTests(unittest.TestCase):
    def test_payload_is_three_digits_of_hundredths(self) -> None:
        self.assertEqual(format_payload(0.1), "010")
        self.assertEqual(format_payload(0.2), "020")
        self.assertEqual(format_payload(0.4), "040")
        self.assertEqual(format_payload(1.0), "100")

    def test_unknown_payload_says_so(self) -> None:
        self.assertEqual(format_payload(None), "unknown")

    def test_learning_rate_keeps_the_exponent_form(self) -> None:
        self.assertEqual(format_learning_rate(1e-4), "1e-4")
        self.assertEqual(format_learning_rate(1e-3), "1e-3")
        self.assertEqual(format_learning_rate(5e-5), "5e-5")

    def test_seeds_join_with_a_dash(self) -> None:
        self.assertEqual(format_seeds([1]), "1")
        self.assertEqual(format_seeds([1, 2, 3]), "1-2-3")


class RoleTests(unittest.TestCase):
    def test_one_seed_is_a_pilot(self) -> None:
        self.assertEqual(run_role([1], 2, 6), "pilot")

    def test_several_seeds_on_a_subset_is_a_stage(self) -> None:
        self.assertEqual(run_role([1, 2, 3], 2, 6), "stage")

    def test_several_seeds_on_every_direction_is_full(self) -> None:
        self.assertEqual(run_role([1, 2, 3], 6, 6), "full")


class NameTests(unittest.TestCase):
    def test_pilot_name(self) -> None:
        self.assertEqual(
            run_name(role="pilot", train_domains=["real"], condition=CONDITION,
                     learning_rate=1e-4, seeds=[1]),
            "pilot-real-suniward040-crop-lr1e-4-s1",
        )

    def test_full_protocol_name(self) -> None:
        self.assertEqual(
            run_name(role="full", train_domains=["real", "ai", "mixed"],
                     condition=CONDITION, learning_rate=1e-4, seeds=[1, 2, 3]),
            "full-real+ai+mixed-suniward040-crop-lr1e-4-s1-2-3",
        )

    def test_the_learning_rate_appears_in_the_name(self) -> None:
        """The two pilots differ only by learning rate, so the names must too."""
        first = run_name(role="pilot", train_domains=["real"], condition=CONDITION,
                         learning_rate=1e-3, seeds=[1])
        second = run_name(role="pilot", train_domains=["real"], condition=CONDITION,
                          learning_rate=1e-4, seeds=[1])
        self.assertNotEqual(first, second)

    def test_an_unreadable_condition_is_visible_in_the_name(self) -> None:
        name = run_name(role="pilot", train_domains=["real"], condition={},
                        learning_rate=1e-4, seeds=[1])
        self.assertIn("unknown", name)


class TagTests(unittest.TestCase):
    def test_every_facet_is_present(self) -> None:
        tags = run_tags(role="stage", detector="srnet", train_domains=["real"],
                        directions=["real_to_real", "real_to_ai"],
                        condition=CONDITION, learning_rate=1e-4, seeds=[1, 2, 3])
        for expected in ("role:stage", "detector:srnet", "algorithm:suniward",
                         "payload:0.4", "preprocess:crop", "lr:1e-4", "seeds:3",
                         "domain:real", "direction:real_to_real",
                         "direction:real_to_ai"):
            self.assertIn(expected, tags)

    def test_user_tags_are_kept_and_not_duplicated(self) -> None:
        tags = run_tags(role="pilot", detector="srnet", train_domains=["real"],
                        directions=["real_to_real"], condition=CONDITION,
                        learning_rate=1e-4, seeds=[1],
                        extra=["rerun", "role:pilot"])
        self.assertIn("rerun", tags)
        self.assertEqual(tags.count("role:pilot"), 1)


class GroupTests(unittest.TestCase):
    def test_group_names_the_cohort_and_condition(self) -> None:
        self.assertEqual(run_group(CONDITION, SHA), "suniward040-crop-f271f135")

    def test_two_cohorts_with_the_same_settings_stay_apart(self) -> None:
        other = "0" * 64
        self.assertNotEqual(run_group(CONDITION, SHA), run_group(CONDITION, other))


class ManifestConditionTests(unittest.TestCase):
    def build_chain(self, directory: Path) -> Path:
        prepared = directory / "cohort_clean.jsonl"
        prepared.write_text("", encoding="utf-8")
        (directory / "cohort_clean.config.json").write_text(
            json.dumps({"name": "cohort", "preprocess_mode": "crop", "size": 256}),
            encoding="utf-8",
        )
        run_dir = directory / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({"algorithm": "suniward", "payload_bpp": 0.4,
                        "record_count": 10, "input_manifest": str(prepared)}),
            encoding="utf-8",
        )
        manifest = run_dir / "dataset_manifest.jsonl"
        manifest.write_text("", encoding="utf-8")
        return manifest

    def test_reads_the_condition_from_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.build_chain(Path(raw))
            condition = manifest_condition(manifest)
        self.assertEqual(condition["algorithm"], "suniward")
        self.assertEqual(condition["payload_bpp"], 0.4)
        self.assertEqual(condition["preprocess_mode"], "crop")

    def test_a_manifest_with_no_run_config_gives_empty_fields(self) -> None:
        """A missing chain must not stop a run, and must not invent a label."""
        with tempfile.TemporaryDirectory() as raw:
            orphan = Path(raw) / "dataset_manifest.jsonl"
            orphan.write_text("", encoding="utf-8")
            condition = manifest_condition(orphan)
        self.assertIsNone(condition["algorithm"])
        self.assertIsNone(condition["preprocess_mode"])


if __name__ == "__main__":
    unittest.main()
